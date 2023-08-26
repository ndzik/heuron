{-# LANGUAGE MultiParamTypeClasses #-}

module Heuron.V2.Backend.Translator where

import Heuron.V2.Network

class Translatable t net where
  type TargetStructure t net
  translate :: net -> t (TargetStructure t net)
