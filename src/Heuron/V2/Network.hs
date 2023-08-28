{-# LANGUAGE UndecidableInstances #-}

module Heuron.V2.Network where

import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.V2.Layer

data Network (b :: Nat) ls where
  (:>:) ::
    (ThrowOnError (CheckValidLayers (Layer i n af op) (Layer i' n' af' op'))) =>
    Layer i n af op ->
    Network b (Layer i' n' af' op' ': ls) ->
    Network b (Layer i n af op ': Layer i' n' af' op' ': ls)
  (:=>) ::
    (ThrowOnError (CheckValidLayers (Layer i n af op) (Layer i' n' af' op'))) =>
    Layer i n af op ->
    Layer i' n' af' op' ->
    Network b '[Layer i n af op, Layer i' n' af' op']

infixr 5 :>:

infixr 6 :=>

type family ThrowOnError (b :: k) :: Constraint where
  ThrowOnError () = ()
  ThrowOnError msg = TypeError msg

type family CheckValidLayers l1 l2 :: ErrorMessage where
  CheckValidLayers (Layer i n af op) (Layer i' n' af' op') = CheckCondition (ValidInputForwarding n i') (MismatchedInputSizeErr n i')

type family CheckCondition (b :: Bool) (msg :: ErrorMessage) :: k where
  CheckCondition 'True _ = ()
  CheckCondition 'False msg = msg

type family ValidInputForwarding n i :: Bool where
  ValidInputForwarding n n = 'True
  ValidInputForwarding _ _ = 'False

type InvalidNetworkConstructionErr i i' n n' af af' op op' =
  ('Text "Invalid network construction: " ':$$: 'Text " >>> [" :<>: 'ShowType (Layer i n af op) :<>: 'Text "] " ':$$: 'Text " incompatible with " ':$$: 'Text " >>> [" :<>: 'ShowType (Layer i' n' af' op')) :<>: 'Text "] "
    :$$: 'Text "Note: Check that your layers have the same batch size and that the expected inputs for each layer match the outputs of its previous layer."

type MismatchedInputSizeErr n i =
  ('Text "Mismatched input size: " ':<>: 'ShowType n ':<>: 'Text " /= " ':<>: 'ShowType i)
    :$$: ('Text "Note: You are trying to pipe the output of a layer with " ':<>: 'ShowType n ':<>: 'Text " neurons into a layer which expects " ':<>: 'ShowType i ':<>: 'Text " inputs.")
