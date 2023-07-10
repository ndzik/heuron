{-# LANGUAGE RankNTypes #-}

module Heuron.V1.Vector where

import Data.Vector (fromList)
import GHC.TypeLits
import Linear.V
import System.Random

-- | Create a Vector with dimension n from a list of values. This will fail if
-- the given list of values does not match the dimension n. For an unsafe
-- variant in IO see `mkV'`.
--
-- Example:
-- @
--  main :: IO ()
--  main = do
--      let _ = mkV @3 [1, 2, 3] -- Just (V3 1.0 2.0 3.0)
--          _ = mkV @3 [1, 2]    -- Nothing
-- @
mkV :: forall n a. (KnownNat n) => [a] -> Maybe (V n a)
mkV = fromVector . fromList

-- | Create a Vector with dimension n from a list of values. This will fail if
-- the given list of values does not match the dimension n.
--
-- Example:
-- @
--  main :: IO ()
--  main = do
--      _ <- mkV' @3 [1, 2, 3] -- (V3 1.0 2.0 3.0)
--      _ <- mkV' @3 [1, 2]    -- error
-- @
mkV' :: forall n a. (KnownNat n) => [a] -> IO (V n a)
mkV' xs = case fromVector . fromList $ xs of
  Just v -> return v
  Nothing -> error "mkVector: failed to create vector"

randomV :: forall n g a. (KnownNat n, RandomGen g) => g -> (V n Double, g)
randomV rng =
  let (v, rng') = random rng
   in (v, rng')

randomV' :: forall n. (KnownNat n) => StdGen -> IO (V n Double)
randomV' rng = do
  let (v, rng') = randomV rng
  setStdGen rng'
  return v
