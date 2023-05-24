{-# LANGUAGE TemplateHaskell #-}

-- | This module provides some miscellaneous functions which can be used to
-- evaluate a Heuron network.
module Heuron.Functions where

import Control.Lens
import System.Random

data Datum = Datum
  { _x :: !Double,
    _y :: !Double,
    _class_ :: !Int
  }
  deriving (Show)

makeLenses ''Datum

spiralData :: Int -> Int -> IO [Datum]
spiralData numOfClasses numOfSamples = do
  rng <- getStdGen
  let alphas = take numOfClasses $ randomRs (0.1 :: Double, 100.0) rng
      betas = take numOfClasses $ randomRs (0.1 :: Double, 1.0) rng
      spiralClasses = map (\(a, b, c) -> mkSpiral a b c) $ zip3 alphas betas [0 .. numOfClasses -1]
      samples = concatMap (\f -> map f [0, 0.08 .. 0.08 * fromIntegral numOfSamples]) spiralClasses
  return samples

mkSpiral :: Double -> Double -> Int -> (Double -> Datum)
mkSpiral = spiral

spiral :: Double -> Double -> Int -> Double -> Datum
spiral alpha beta cl phi = Datum x y cl
  where
    x = radius * cos phi
    y = radius * sin phi
    -- Logarithmic spiral:
    --  alpha > 0
    --  beta != 0
    radius = alpha * e ** (beta * phi)
    e = exp 1
