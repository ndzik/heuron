{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE TypeFamilies #-}

module Main where

main :: IO ()
main = do
  -- We want correct by construction networks. This means they have to be
  -- lifted to the typesystem. Using type constraints we will guarantee only
  -- valid networks to be described.
  --
  -- The networks constructed are only a description and its interpretation is
  -- up to some interpreter which understands said typelifted DSL and generates
  -- some code which can be used to train/run the network.
  --
  -- Each network has an input, optional hidden and an output layer.
  -- The layers have to be compatible.
  -- Each layer consists of neurons:
  --  * Neuron:
  --    - The inputs for a layer can be of arbitrary dimensions.
  --    - Has weight for each input.
  --    - Has an adjustable bias.
  --    - Has an activation function of some kind, which receives the bias
  --      added to the sum of all inputs multiplied by their corresponding
  --      weights.
  --
  -- As contained in the description of network, the described network is, in
  -- essence, a graph which can be constructed at compile time.
  --
  -- Possible DSL:
  -- Input i activation :> Hidden n activation :> Hidden m activation' :> Output x classifier
  --
  -- It should not be possible to have the following types considered to be
  -- valid networks:
  --  Input i a :> Input i' a' -- | malformed network
  --  Output x classifier      -- | This is not a valid network, previous layers
  --                                are missing.
  --  Hidden n a :> Input i a' -- | This breaks dataflow
  --  Hidden n a               -- | Again, not valid network, input missing.
  --
  -- With the above constraints in mind we can describe a network statically:
  --  Network (Input i activation) (Output x classifier) <Hidden-Layer-Description>
  --
  -- The hidden layer description will utilize the infix type-constructor `:>:`
  --  Hidden j a :>: Hidden k a' :>: Hidden l a''
  print "heuron"
