{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE DeriveFunctor #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE InstanceSigs #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TemplateHaskell #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeOperators #-}

module Main where

import Control.Lens
import Control.Monad.State
import Data.Vector (fromList)
import Diagrams.Backend.SVG.CmdLine
import GHC.TypeLits
import Heuron.Functions
import qualified Heuron.V1 as Heuron
import Linear.Matrix
import Linear.Metric
import Linear.V
import Linear.Vector
import Plots

main :: IO ()
main = return ()

type ActivationFunction = Double -> Double

newtype Layer (n :: k) (m :: k) a = Layer {unlayer :: State (LayerState n m) a}
  deriving (Functor, Applicative, Monad, MonadState (LayerState n m))

-- | Layers state, where n is the number of neurons and m is the number of
-- inputs.
data LayerState (i :: k) (n :: k) = LayerState
  { -- | Weights of the layer as a matrix of size m x n, where n is the number
    -- of neurons and m is the number of inputs. Each neuron is identified by
    -- the row index.
    _weights :: !(V n (V i Double)),
    -- | Bias of the layer as a vector of size n, where n is the number of
    -- neurons.
    _bias :: !(V n Double),
    -- | The activation function used for each neuron in the layer.
    _activation :: !ActivationFunction
  }

makeLenses ''LayerState

relu :: ActivationFunction
relu = max 0

data Network a = Network
  {
  }

neuralNet :: [Datum] -> IO ()
neuralNet ds = do
  input <- mapM (\d -> mkV @2 [d ^. x, d ^. y]) ds

  -- Input layer with 2 inputs and 3 neurons.
  inputLayerWeights <- mkM @2 @3 [[1 | _ <- [1 .. 3]] | _ <- [1 .. 2]]
  inputLayerBias <- mkV @3 [1 | _ <- [1 .. 3]]

  hidden01LayerWeights <- mkM @3 @4 undefined
  hidden01LayerBias <- mkV @4 undefined
  hidden02LayerWeights <- mkM @4 @4 undefined
  hidden02LayerBias <- mkV @4 undefined
  outputLayerWeights <- mkM @4 @3 undefined
  outputLayerBias <- mkV @3 undefined
  --
  ---
  let o1 = LayerState inputLayerWeights inputLayerBias relu
      o2 = LayerState hidden01LayerWeights hidden01LayerBias relu
      o3 = LayerState hidden02LayerWeights hidden02LayerBias relu
      o4 = LayerState outputLayerWeights outputLayerBias relu
      xyz = o1 :>: o2 :>: o3 :>: o4 :>: HiddenEnd
  ---
  -- let output1 = evalLayer (LayerState inputLayerWeights inputLayerBias relu) (forwardInput . head $ input)
  --     output2 = evalLayer (LayerState hidden01LayerWeights hidden01LayerBias relu) (forwardInput . fst $ output1)
  --     output3 = evalLayer (LayerState hidden02LayerWeights hidden02LayerBias relu) (forwardInput . fst $ output2)
  --     output4 = evalLayer (LayerState outputLayerWeights outputLayerBias relu) (forwardInput . fst $ output3)
  ---
  return ()

evalLayer :: LayerState n m -> Layer n m a -> (a, LayerState n m)
evalLayer s action = runState (unlayer action) s

-- trainOn input = forward input >>= backpropagate >>= optimize

-- forward' input = do
--   il <- use inputLayer
--   forwardInput input il >>= forwardHidden >>= forwardOutput

class Forward a where
  forward :: (KnownNat n, KnownNat i) => V i Double -> Layer i n (V n Double)

infixr 5 :>:

data Hidden as where
  HiddenEnd :: Hidden '[]
  (:>:) :: a -> Hidden as -> Hidden (a ': as)

instance Forward (Layer i n) where
  forward input = get <&> forwardInput input

-- instance (Forward as, m ~ j) => Forward (Hidden (LayerState n m ': LayerState j k ': as)) where
--   forward input = forward @as . forward @(LayerState n m) $ input

-- [a, b, c, d]
-- input => forward a input & forward b & forward c & forward d

backpropagate = undefined

optimize = undefined

forwardInput :: (KnownNat n, KnownNat i) => V i Double -> LayerState i n -> V n Double
forwardInput inputs s = do
  -- This does for every neuron i (row) in the layer:
  -- > Σ(w_ij * x_j) + b_i
  -- where `i` is the neuron index and `j` is the input index.
  let weightedInputs = ((s ^. weights) !* inputs) ^+^ (s ^. bias)
      activationFunction = s ^. activation
  -- Apply activation and return result for this layer.
  activationFunction <$> weightedInputs

-- forwardInput :: (KnownNat n, KnownNat m) => V m Double -> Layer n m (V n Double)
-- forwardInput inputs = do
--   -- This does for every neuron i (row) in the layer:
--   -- > Σ(w_ij * x_j) + b_i
--   -- where `i` is the neuron index and `j` is the input index.
--   weightedInputs <- (use weights <&> (!* inputs) <&> (^+^)) <*> use bias
--   -- Apply activation and return result for this layer.
--   activationFunction <- use activation
--   return $ fmap activationFunction weightedInputs

mkV :: forall n a. (KnownNat n) => [a] -> IO (V n a)
mkV xs = case fromVector . fromList $ xs of
  Just v -> return v
  Nothing -> error "mkVector: failed to create vector"

mkM :: forall i n. (KnownNat i, KnownNat n) => [[Double]] -> IO (V n (V i Double))
mkM xs = mapM (mkV @i) xs >>= mkV @n
