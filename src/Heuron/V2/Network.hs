{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.V2.Network where

import Data.Kind (Constraint)
import GHC.TypeLits
import Heuron.V2.Layer

type family (xs :: [k]) <++> (ys :: [k]) :: [k] where
  '[] <++> ys = ys
  (x ': xs) <++> ys = x ': (xs <++> ys)

data Network (b :: Nat) ls where
  (:+:) :: (CheckCompatibleBoundary ls ls') => Network b ls -> Network b ls' -> Network b (ls <++> ls')
  (:>:) ::
    (CheckValidLayers (Layer b i o l) (Layer b i' o' l'), o ~ i') =>
    Layer b i o l ->
    Network b (Layer b i' o' l' ': ls) ->
    Network b (Layer b i o l ': Layer b i' o' l' ': ls)
  (:=>) ::
    (CheckValidLayers (Layer b i o l) (Layer b i' o' l'), o ~ i') =>
    Layer b i o l ->
    Layer b i' o' l' ->
    Network b '[Layer b i o l, Layer b i' o' l']

infixr 5 :>:

infixr 6 :=>

type CheckCompatibleBoundary :: [*] -> [*] -> Constraint
type family CheckCompatibleBoundary ls ls' where
  CheckCompatibleBoundary '[l] (l' ': ls') = CheckValidLayers l l'
  CheckCompatibleBoundary (l ': ls) ls' = CheckCompatibleBoundary ls ls'

type CheckValidLayers :: * -> * -> Constraint
type family CheckValidLayers l1 l2 where
  CheckValidLayers (Layer b i o l) (Layer b i' o' l') = CheckCondition (ValidInputForwarding o i') (MismatchedInputSizeErr o i')

type InputLayerOfNetwork :: [*] -> *
type family InputLayerOfNetwork ls where
  InputLayerOfNetwork '[Layer b i n l] = Layer b i n l
  InputLayerOfNetwork (Layer b i n l ': ls) = Layer b i n l
  InputLayerOfNetwork ls = TypeError ('Text "Incompatible type to determine input layer of a network")

type OutputLayerOfNetwork :: [*] -> *
type family OutputLayerOfNetwork ls where
  OutputLayerOfNetwork '[Layer b i n l] = Layer b i n l
  OutputLayerOfNetwork (Layer b i n l ': ls) = OutputLayerOfNetwork ls
  OutputLayerOfNetwork ls = TypeError ('Text "Determining output layer for ill-formed network: " ':$$: 'ShowType ls)

type CheckCondition :: Bool -> ErrorMessage -> Constraint
type family CheckCondition (b :: Bool) msg where
  CheckCondition 'True _ = ()
  CheckCondition 'False msg = TypeError msg

type ValidInputForwarding :: [Nat] -> [Nat] -> Bool
type family ValidInputForwarding n i :: Bool where
  ValidInputForwarding n n = 'True
  ValidInputForwarding _ _ = 'False

type InvalidNetworkConstructionErr b i i' n n' =
  ('Text "Invalid network construction: " ':$$: 'Text " >>> [" :<>: 'ShowType (Layer b i n) :<>: 'Text "] " ':$$: 'Text " incompatible with " ':$$: 'Text " >>> [" :<>: 'ShowType (Layer b i' n')) :<>: 'Text "] "
    :$$: 'Text "Note: Check that your layers have the same batch size and that the expected inputs for each layer match the outputs of its previous layer."

type MismatchedInputSizeErr n i =
  ('Text "Mismatched input size: " ':<>: 'ShowType n ':<>: 'Text " /= " ':<>: 'ShowType i)
    :$$: ('Text "Note: You are trying to pipe the output of a layer with " ':<>: 'ShowType n ':<>: 'Text " neurons into a layer which expects " ':<>: 'ShowType i ':<>: 'Text " inputs.")
