{-# LANGUAGE UndecidableInstances #-}

module Heuron.V2.Network where

import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.V2.Layer

data Network (b :: Nat) ls where
  (:>:) ::
    (CheckValidLayers (Layer i n af op) (Layer i' n' af' op')) =>
    Layer i n af op ->
    Network b (Layer i' n' af' op' ': ls) ->
    Network b (Layer i n af op ': Layer i' n' af' op' ': ls)
  (:=>) ::
    (CheckValidLayers (Layer i n af op) (Layer i' n' af' op')) =>
    Layer i n af op ->
    Layer i' n' af' op' ->
    Network b '[Layer i n af op, Layer i' n' af' op']

infixr 5 :>:

infixr 6 :=>

type family CheckValidLayers l1 l2 :: Constraint where
  CheckValidLayers (Layer i n af op) (Layer n n' af' op') = ()
  CheckValidLayers (Layer i n af op) (Layer i' n' af' op') = TypeError (MismatchedInputSizeErr n i')

type family NotEqualNat (n :: k) (n' :: k) :: Bool where
  NotEqualNat n n = 'True
  NotEqualNat n n' = 'False

type family If (b :: Bool) (t :: k) (f :: k) :: k where
  If 'True t f = t
  If 'False t f = f

type InvalidNetworkConstructionErr i i' n n' af af' op op' =
  ('Text "Invalid network construction: " ':$$: 'Text " >>> [" :<>: 'ShowType (Layer i n af op) :<>: 'Text "] " ':$$: 'Text " incompatible with " ':$$: 'Text " >>> [" :<>: 'ShowType (Layer i' n' af' op')) :<>: 'Text "] "
    :$$: 'Text "Note: Check that your layers have the same batch size and that the expected inputs for each layer match the outputs of its previous layer."

type MismatchedBatchSizeErr b b' =
  ('Text "Mismatched batch size: " ':<>: 'ShowType b ':<>: 'Text " /= " ':<>: 'ShowType b')
    :$$: 'Text "Note: You are trying to create a network with layers expecting different batch sizes."

type MismatchedInputSizeErr n i =
  ('Text "Mismatched input size: " ':<>: 'ShowType n ':<>: 'Text " /= " ':<>: 'ShowType i)
    :$$: ('Text "Note: You are trying to pipe the output of a layer with " ':<>: 'ShowType n ':<>: 'Text " neurons into a layer which expects " ':<>: 'ShowType i ':<>: 'Text " inputs.")
