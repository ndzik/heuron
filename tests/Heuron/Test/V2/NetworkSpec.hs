{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE TypeOperators #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.Test.V2.NetworkSpec where

import Data.Kind (Constraint)
import GHC.TypeLits (ErrorMessage (..), TypeError)
import Heuron.V2.Layer
import Heuron.V2.Network

compileTimeCheck ::
  () =>
  IO ()
compileTimeCheck = do
  let invalidNetworkConstruction ::
        ( ExpectError
            (MismatchedInputSizeErr 11 10)
            (CheckValidLayers (Layer 10 11 Double Double) (Layer 10 10 Double Double))
        ) =>
        ()
      invalidNetworkConstruction = ()

  let tests =
        [ invalidNetworkConstruction
        ]
  return ()

type family ExpectError (expected :: ErrorMessage) (actual :: ErrorMessage) :: Constraint where
  ExpectError expected expected = ()
  ExpectError expected actual = TypeError ('Text "Expected error: " ':<>: expected ':$$: 'Text "Actual error: " ':<>: actual)
