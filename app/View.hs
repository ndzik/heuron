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
            [ vstack [label "Network" `styleBasic` [textSize 32], openGLWidget $ hm ^. heuronModelNet]
            ],
          vstack
            [ label "Metadata",
              spacer,
              hgrid
                [ vstack [flip styleBasic [textSize 12] . label . pack $ printf "Epoch: %d/%d" (hm ^. heuronModelCurrentEpoch) (hm ^. heuronModelMaxEpochs)] `styleBasic` [bgColor (rgbHex "#2c2d2e")],
                  spacer `styleBasic` [bgColor bgCol],
                  vstack [flip styleBasic [textSize 12] . label . pack . printf "Loss: %.4f" $ hm ^. heuronModelAvgLoss] `styleBasic` [bgColor (rgbHex "#2c2d2e")],
                  spacer `styleBasic` [bgColor bgCol],
                  vstack [flip styleBasic [textSize 12] . label . pack . printf "Accuracy: %.3f" $ hm ^. heuronModelAccuracy] `styleBasic` [bgColor (rgbHex "#2c2d2e")]
                ]
            ]
        ]
        `styleBasic` [padding 8, bgColor bgCol]
    bgCol = rgbHex "#181a1b"
