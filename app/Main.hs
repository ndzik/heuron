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
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Control.Lens
import Control.Monad.State
import Data.Vector (fromList)
import Diagrams.Backend.SVG.CmdLine
import GHC.TypeLits
import Heuron.Functions
import qualified Heuron.V1 as Heuron
import Heuron.V1.Network
import Linear.Matrix
import Linear.Metric
import Linear.V
import Linear.Vector
import Plots

main :: IO ()
main = return ()

relu :: ActivationFunction
relu = max 0

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

  let o1 = Layer inputLayerWeights inputLayerBias relu
      o2 = Layer hidden01LayerWeights hidden01LayerBias relu
      o3 = Layer hidden02LayerWeights hidden02LayerBias relu
      o4 = Layer outputLayerWeights outputLayerBias relu
      network = o1 :>: o2 :>: o3 :>: o4 :>: NetworkEnd
      result = forward network (head input)
  return ()

mkV :: forall n a. (KnownNat n) => [a] -> IO (V n a)
mkV xs = case fromVector . fromList $ xs of
  Just v -> return v
  Nothing -> error "mkVector: failed to create vector"

mkM :: forall i n. (KnownNat i, KnownNat n) => [[Double]] -> IO (V n (V i Double))
mkM xs = mapM (mkV @i) xs >>= mkV @n
