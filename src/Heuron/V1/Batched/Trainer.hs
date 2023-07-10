{-# LANGUAGE ConstraintKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}

module Heuron.V1.Batched.Trainer where

import Control.Lens
import Control.Monad.State
import GHC.TypeLits
import Heuron.V1.Batched.Backprop (Backprop (backprop))
import Heuron.V1.Batched.Forward (Forward (forward))
import Heuron.V1.Batched.Input (Input)
import Heuron.V1.Batched.Loss (LossComparator (..))
import Heuron.V1.Batched.Network

newtype Trainer (b :: Nat) net l m a = Trainer {_train :: StateT (TrainerState b net l) m a}
  deriving (Functor, Applicative, Monad, MonadState (TrainerState b net l))

data TrainerState (b :: Nat) net l = TrainerState
  { _network :: !(Network b net),
    _lossifier :: !l
  }

makeLenses ''TrainerState

runTrainer :: Trainer b net l m a -> TrainerState b net l -> m (a, TrainerState b net l)
runTrainer trainer = runStateT (_train trainer)

type TrainerConstraints b net l m =
  ( Monad m,
    LossComparator l,
    KnownNat b,
    Forward (Network b net),
    Backprop (Network b net)
  )

oneEpoch ::
  ( TrainerConstraints b net l m,
    InputOf (Network b net) ~ Input b n Double,
    FinalOutputOf (Network b net) ~ Input b n' Double,
    Input b n' Double ~ NextOutput (Reversed (Network b net)),
    KnownNat n,
    KnownNat n',
    n ~ n'
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m ()
runEpoch input truth = trainForward input >>= trainBackprop truth

trainForward ::
  ( TrainerConstraints b net l m,
    InputOf (Network b net) ~ Input b n Double
  ) =>
  Input b n Double ->
  Trainer b net l m (FinalOutputOf (Network b net))
trainForward input =
  use network >>= \n -> do
    let (n', forwardResult) = forward n input
    network .= n'
    return forwardResult

trainBackprop ::
  ( TrainerConstraints b net l m,
    KnownNat n,
    FinalOutputOf (Network b net) ~ Input b n Double,
    Input b n Double ~ NextOutput (Reversed (Network b net))
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m ()
trainBackprop truth forwardResult =
  use network >>= \n -> do
    loss <- trainLoss truth forwardResult
    let n' = backprop n forwardResult loss
    network .= n'

-- | trainLoss computes the loss of the network compared to the truth for each
-- sample in a batch.
trainLoss ::
  forall n n' b net l m.
  ( Monad m,
    LossComparator l,
    KnownNat n,
    KnownNat b
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m (Input b n Double)
trainLoss truth forwardResult = use (lossifier . to (losser @_ @b @n)) <*> pure truth <*> pure forwardResult

-- | trainAccuracy computes the average accuracy of the network for the given
-- batch of samples.
trainAccuracy ::
  forall n n' b net l m.
  ( TrainerConstraints b net l m,
    InputOf (Network b net) ~ Input b n Double,
    KnownNat n
  ) =>
  Input b n Double ->
  Input b n Double ->
  Trainer b net l m Double
trainAccuracy truth forwardResult = use (lossifier . to (accurator @_ @b @n)) <*> pure truth <*> pure forwardResult
