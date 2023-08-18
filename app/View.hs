{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}

module View where

import Control.Lens
import Data.Text (Text, pack)
import GHC.TypeLits (KnownNat, Nat)
import Heuron.V1
import Heuron.V1.Batched
import Heuron.V1.Batched.Network
import Monomer
import OpenGL
import Text.Printf (printf)
import Types

buildUI :: WidgetEnv HeuronModel HeuronEvent -> HeuronModel -> WidgetNode HeuronModel HeuronEvent
buildUI we hm = widgetTree
  where
    widgetTree =
      vstack
        [ hgrid
            [ vstack [label "Network", openGLWidget $ hm ^. heuronModelNet]
            ],
          vstack
            [ label "Metadata",
              hgrid
                [ vstack [label . pack $ printf "Epoch: %d/%d" (hm ^. heuronModelCurrentEpoch) (hm ^. heuronModelMaxEpochs)],
                  vstack [label . pack . printf "Loss: %.4f" $ hm ^. heuronModelAvgLoss],
                  vstack [label . pack . printf "Accuracy: %.3f" $ hm ^. heuronModelAccuracy]
                ]
            ]
        ]
