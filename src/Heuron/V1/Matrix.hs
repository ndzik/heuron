{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module Heuron.V1.Matrix where

import Control.Monad (join)
import GHC.TypeLits
import Heuron.V1.Vector
import Linear.V

-- | Creates a matrix with dimension n x m from a list of lists of values. This
-- will fail if the given list of lists of values does not match the dimension
-- n x m.
--
-- Example:
-- @
--  main :: IO ()
--  main = do
--     let _ = mkM @3 @2 [[1, 2], [3, 4], [5, 6]] -- Just (V3 (V2 1.0 2.0)
--                                                --          (V2 3.0 4.0)
--                                                --          (V2 5.0 6.0))
--         _ <- mkM @3 @2 [[1, 2], [3, 4], [5]]   -- Nothing
-- @
mkM :: forall n m. (KnownNat n, KnownNat m) => [[Double]] -> Maybe (V n (V m Double))
mkM = join . mapM (mkV @n) . mapM (mkV @m)

-- | Creates a matrix with dimension n x m from a list of lists of values. This
-- will fail if the given list of lists of values does not match the dimension
-- n x m.
--
-- Example:
-- @
--  main :: IO ()
--  main = do
--     _ <- mkM @3 @2 [[1, 2], [3, 4], [5, 6]] -- (V3 (V2 1.0 2.0)
--                                             --     (V2 3.0 4.0)
--                                             --     (V2 5.0 6.0))
--     _ <- mkM @3 @2 [[1, 2], [3, 4], [5]]    -- error
-- @
mkM' :: forall n m. (KnownNat n, KnownNat m) => [[Double]] -> IO (V n (V m Double))
mkM' xs = mapM (mkV' @m) xs >>= mkV' @n
