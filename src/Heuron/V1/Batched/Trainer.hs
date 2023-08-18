{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Heuron.V1.Batched.Trainer where

import Control.Lens
import Control.Monad.State
import Data.Data (Proxy (..))
import GHC.TypeLits
import Heuron.V1.Batched.Activation (Differentiable (DerivativeReturn, derivative))
import Heuron.V1.Batched.Backprop (Backprop (backprop))
import Heuron.V1.Batched.Forward (Forward (forward))
import Heuron.V1.Batched.Input (Input)
import Heuron.V1.Batched.Loss (LossComparator (..))
import Heuron.V1.Batched.Network
import Heuron.V1.Matrix
import Linear (sumV)
import Text.Printf (printf)

newtype Trainer (b :: Nat) net l m a = Trainer {_trainWith :: StateT (TrainerState b net l) m a}
  deriving (Functor, Applicative, Monad, MonadState (TrainerState b net l), MonadIO)

data TrainerState (b :: Nat) net l = TrainerState
  { _network :: !(Network b net),
    _lossifier :: !l
  }

makeLenses ''TrainerState

data TrainingResult = TrainingResult
  { _trainingResultLoss :: !Double,
    _trainingResultAccuracy :: !Double
  }
  deriving (Show)

makeLenses ''TrainingResult

runTrainer :: Trainer b net l m a -> TrainerState b net l -> m (a, TrainerState b net l)
runTrainer trainer = runStateT (_trainWith trainer)

type TrainerConstraints b net l m =
  ( Monad m,
    LossComparator l,
    KnownNat b,
    Forward (Network b net),
    Backprop (Network b net)
  )

-- | oneEpoch trains the network for one epoch, using the given input and
-- expected output. Returns the accuracy of the network for the given batch.
oneEpoch ::
  ( TrainerConstraints b net l m,
    InputOf (Network b net) ~ Input b n Double,
    FinalOutputOf (Network b net) ~ Input b n' Double,
    DerivativeReturn l (Input b n' Double) ~ Input b n' Double,
    Input b n' Double ~ NextOutput (Reversed (Network b net)),
    KnownNat n,
    KnownNat n'
  ) =>
  Input b n Double ->
  Input b n' Double ->
  Trainer b net l m TrainingResult
oneEpoch input truth = trainForward input >>= trainBackprop truth

debugOneEpoch ::
  ( TrainerConstraints b net l m,
    MonadIO m,
    Showable net,
    InputOf (Network b net) ~ Input b n Double,
    DerivativeReturn l (Input b n' Double) ~ Input b n' Double,
    FinalOutputOf (Network b net) ~ Input b n' Double,
    Input b n' Double ~ NextOutput (Reversed (Network b net)),
    KnownNat n,
    KnownNat n'
  ) =>
  Input b n Double ->
  Input b n' Double ->
  Trainer b net l m Double
debugOneEpoch input truth = do
  forwardResult <- trainForward input
  use network >>= liftIO . putStrLn . printf "Network:\n%s" . show
  liftIO . putStrLn $ printf "ForwardResult:\n%s" (prettyMatrix forwardResult)
  liftIO . putStrLn $ printf "Truth:\n%s" (prettyMatrix truth)
  backpropResult <- debugTrainBackprop truth forwardResult
  liftIO . putStrLn $ printf "BackpropResult:\n%.4f" backpropResult
  return backpropResult

debugTrainBackprop ::
  ( TrainerConstraints b net l m,
    MonadIO m,
    KnownNat n,
    KnownNat n',
    DerivativeReturn l (Input b n' Double) ~ Input b n' Double,
    InputOf (Network b net) ~ Input b n Double,
    FinalOutputOf (Network b net) ~ Input b n' Double,
    Input b n' Double ~ NextOutput (Reversed (Network b net))
  ) =>
  Input b n' Double ->
  Input b n' Double ->
  Trainer b net l m Double
debugTrainBackprop truth forwardResult = do
  use network >>= \n -> do
    loss <- trainLoss truth forwardResult
    liftIO . putStrLn $ printf "Loss:\n%s" (prettyMatrix loss)
    lossGradient <- trainLossGradient truth forwardResult
    liftIO . putStrLn $ printf "LossGradient:\n%s" (prettyMatrix lossGradient)
    let n' = backprop n forwardResult lossGradient
    network .= n'
  trainAccuracy truth forwardResult

trainForward ::
  ( TrainerConstraints b net l m,
    InputOf (Network b net) ~ Input b n Double,
    FinalOutputOf (Network b net) ~ Input b n' Double
  ) =>
  Input b n Double ->
  Trainer b net l m (Input b n' Double)
trainForward input =
  use network >>= \n -> do
    let (n', forwardResult) = forward n input
    network .= n'
    return forwardResult

trainBackprop ::
  forall b n n' net l m.
  ( TrainerConstraints b net l m,
    KnownNat b,
    KnownNat n,
    KnownNat n',
    DerivativeReturn l (Input b n' Double) ~ Input b n' Double,
    InputOf (Network b net) ~ Input b n Double,
    FinalOutputOf (Network b net) ~ Input b n' Double,
    Input b n' Double ~ NextOutput (Reversed (Network b net))
  ) =>
  Input b n' Double ->
  Input b n' Double ->
  Trainer b net l m TrainingResult
trainBackprop truth forwardResult = do
  use network >>= \n -> do
    lossGradient <- trainLossGradient truth forwardResult
    let n' = backprop n forwardResult lossGradient
    network .= n'
    l <-
      trainLoss truth forwardResult >>= \l -> do
        let batchSize = fromIntegral $ natVal (Proxy @b)
            allLoss = sum . sumV $ l
            avgLoss = allLoss / batchSize
        return avgLoss
    acc <- trainAccuracy truth forwardResult
    return $ TrainingResult l acc

-- | trainLoss computes the loss of the network compared to the truth
-- for each sample in a batch.
trainLoss ::
  forall n b net l m.
  ( Monad m,
    LossComparator l,
    FinalOutputOf (Network b net) ~ Input b n Double,
    KnownNat n,
    KnownNat b
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m (Input b n Double)
trainLoss truth forwardResult = use (lossifier . to (losser @_ @b @n)) <*> pure truth <*> pure forwardResult

-- | trainLossGradient computes the loss gradients of the network compared to the truth
-- for each sample in a batch.
trainLossGradient ::
  forall n b net l m.
  ( Monad m,
    LossComparator l,
    FinalOutputOf (Network b net) ~ Input b n Double,
    DerivativeReturn l (Input b n Double) ~ Input b n Double,
    KnownNat n,
    KnownNat b
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m (Input b n Double)
trainLossGradient truth forwardResult = use (lossifier . to (derivative @_ @b @n)) <*> pure truth <*> pure forwardResult

-- | trainAccuracy computes the average accuracy of the network for the given
-- batch of samples.
trainAccuracy ::
  forall n b net l m.
  ( TrainerConstraints b net l m,
    FinalOutputOf (Network b net) ~ Input b n Double,
    KnownNat n
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m Double
trainAccuracy truth forwardResult = use (lossifier . to (accurator @_ @b @n)) <*> pure truth <*> pure forwardResult
