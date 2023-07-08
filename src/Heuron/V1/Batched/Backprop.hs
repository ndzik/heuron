{-# LANGUAGE UndecidableInstances #-}

module Heuron.V1.Batched.Backprop where

import Control.Lens
import Data.Kind (Constraint)
import GHC.TypeLits (KnownNat)
import Heuron.Functions
import Heuron.V1.Batched.Activation
import Heuron.V1.Batched.Input
import Heuron.V1.Batched.Layer
import Heuron.V1.Batched.Network
import Heuron.V1.Batched.Optimizer
import Linear (Metric (..))
import Linear.Matrix
import Linear.V (Dim, V)
import Linear.Vector hiding (zero)

-- | Backprop is the typelevel recursion entry for backpropagating a network.
-- Backprop can only be applied to Networks whose reversed list of layers
-- implement the `Backprop'` typeclass.
-- This is trivially the case if the Network is `Compatible` and implements
-- the `Forward` typeclass.
class Backprop as where
  backprop :: (o ~ NextOutput (Reversed as)) => as -> o -> o -> as

instance
  ( Backprop' (Network b sl),
    Reversed' sl '[] ~ ls,
    Reversed' ls '[] ~ sl,
    IsReversed sl '[] ls,
    IsReversed ls '[] sl
  ) =>
  Backprop (Network b ls)
  where
  backprop ann i = reverseNetwork . backprop' (reverseNetwork ann) i

-- | Backprop' is the typelevel interpreter for our constructed network. It will
-- generate the correct amounts of backprop calls for each layer constructing
-- a network where all layers contain updated weights and biases.
class Backprop' as where
  backprop' :: (o ~ NextOutput as) => as -> o -> o -> as

instance
  {-# OVERLAPPING #-}
  ( Backpropagatable (Network b [Layer b' i n af op, Layer b'' i' n' af' op']),
    KnownNat i,
    KnownNat i',
    KnownNat b,
    KnownNat b',
    KnownNat b'',
    KnownNat n
  ) =>
  Backprop' (Network b [Layer b' i n af op, Layer b'' i' n' af' op'])
  where
  backprop' (l1 :>: l2 :>: NetworkEnd) l1Output gradients =
    let (l1', gradients') = backpropLayer l1 l1Output gradients
        l2Output = l1 ^. input
        (l2', _) = backpropLayer l2 l2Output gradients'
     in l1' :>: l2' :>: NetworkEnd

instance
  ( Backpropagatable (Network b (Layer b' i n af op : Layer b'' i' i af' op' : Layer b''' i'' i' af'' op'' : rs)),
    KnownNat i,
    KnownNat b,
    KnownNat b',
    KnownNat b'',
    KnownNat b''',
    KnownNat n,
    Backprop' (Network b (Layer b' i' i af' op' : Layer b'' i'' i' af'' op'' : rs))
  ) =>
  Backprop' (Network b (Layer b' i n af op : Layer b'' i' i af' op' : Layer b''' i'' i' af'' op'' : rs))
  where
  backprop' (l1 :>: l2 :>: ls) l1Output gradients =
    let (l1', gradients') = backpropLayer l1 l1Output gradients
        l2Output = l1 ^. input
     in l1' :>: backprop' (l2 :>: ls) l2Output gradients'

-- | Backpropagatable checks that a given network is backpropagatable. It is
-- similar to the Compatibility check in `Heuron.V1.Batched.Network`.
type family Backpropagatable a :: Constraint where
  Backpropagatable (Network b '[Layer b' i n af op, Layer b'' i' n' af' op']) =
    ( i ~ n',
      b ~ b',
      b ~ b'',
      ActivationFunction af,
      ActivationFunction af',
      Optimizable (Layer b' i n af op),
      Optimizable (Layer b'' i' n' af' op')
    )
  Backpropagatable (Network b (Layer b' i n af op : Layer b'' i' n' af' op' : Layer b''' i'' n'' af'' op'' : ls)) =
    ( i ~ n',
      b ~ b',
      b ~ b'',
      b ~ b''',
      ActivationFunction af,
      ActivationFunction af',
      ActivationFunction af'',
      Optimizable (Layer b' i n af op),
      Optimizable (Layer b'' i' n' af' op'),
      Optimizable (Layer b''' i'' n'' af'' op''),
      Backpropagatable (Network b (Layer b i' n' af' op' : Layer b''' i'' n'' af'' op'' : ls))
    )

backpropLayer ::
  ( KnownNat n,
    KnownNat i,
    KnownNat b,
    ActivationFunction af,
    Optimizable (Layer b i n af op)
  ) =>
  -- | Layer to backpropagate.
  Layer b i n af op ->
  -- | Output of the layer that is backpropagated.
  Input b n Double ->
  -- | Gradients of the previous layer.
  Input b n Double ->
  -- | Layer with updated weights and biases and the gradients used for the
  -- previous layer.
  (Layer b i n af op, Input b i Double)
backpropLayer l originalOutput prevGradients =
  let dActivation = l ^. activationFunction . to derivative
      ws = l ^. weights
      is = l ^. input
      bs = l ^. bias
      weightedInput = (is !*! transpose ws) <&> (+ bs)
      activated = dActivation weightedInput originalOutput
      gradients = mergeEntriesWith (*) activated prevGradients
   in -- Influence of the inputs to this layer on the output of this layer.
      --
      -- Example:
      -- l     k
      -- o
      --  \____o
      --  /\ /
      -- o  /
      --  \/_\_o
      --  /
      -- o
      --
      -- l = Layer s 3
      -- k = Layer 3 2
      --
      -- For a single observation, k will have a matrix M 2 3:
      --  A matrix where each row describes the derivates for the associated
      --  neuron in k and it's corresponding inputs.
      --  Collapsing the matrix to M 1 3 = V 3 will result in a gradient vector
      --  that describes the cumulative effect on the global error of a neuron
      --  from the previous layer.
      -- For a batch of observations of size b, k will have a list of matrices
      -- V b (M 2 3):
      --  Again collapsing the inner matrix for each observation into a
      --  cumulative gradient will give the net influence of the neurons from
      --  the previous layers on the network error. We keep the samples of the
      --  dimensions: V b (M 2 3) =collapse=> V b (M 1 3) = V b (V 3) = M b 3.
      --
      -- l has weights (M 3 s)
      -- k returns gradient (M b 3)
      let dInputs = gradients !*! ws
          dWeights = transpose gradients !*! is
          dBias = foldr (^+^) zero gradients
          optimizedLayer = optimize l dWeights dBias
       in (optimizedLayer, dInputs)
