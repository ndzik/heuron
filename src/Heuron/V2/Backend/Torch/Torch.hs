{-# LANGUAGE GeneralizedNewtypeDeriving #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE OverloadedStrings #-}

module Heuron.V2.Backend.Torch.Torch (runPyTorch, saveToModule) where

import Control.Monad.State
import Data.Proxy
import Data.Text (Text)
import qualified Data.Text as T
import qualified Data.Text.IO as T
import GHC.TypeLits
import Heuron.V1.Batched.Activation
import Heuron.V2.Backend.Translator
import qualified Heuron.V2.Drop as Drop
import Heuron.V2.Layer
import Heuron.V2.Network
import qualified Heuron.V2.Residual as Residual
import Text.Printf

newtype PyTorchGen a = PyTorchGen {unGen :: State PyTorchState a}
  deriving (Functor, Applicative, Monad, MonadState PyTorchState)

data PyTorchState = PyTorchState
  { layerDefs :: [Text],
    forwardDefs :: [Text],
    submodules :: [Text],
    counter :: Int
  }

saveToModule :: FilePath -> Text -> IO ()
saveToModule = T.writeFile

runPyTorch :: (Translatable PyTorchGen (Network b ls)) => Network b ls -> Text
runPyTorch net = renderModule s
  where
    initialState = PyTorchState [] [] [] 0
    s = execState (unGen (translate net)) initialState

renderModule :: PyTorchState -> Text
renderModule (PyTorchState initLines forwardLines submodules _) =
  T.unlines $
    [ "import torch",
      "import torch.nn as nn",
      "",
      ""
    ]
      ++ submodules
      ++ [ "",
           "class Model(nn.Module):",
           "    def __init__(self):",
           "        super().__init__()"
         ]
      ++ indent initLines
      ++ [ "",
           "    def forward(self, x):"
         ]
      ++ indent forwardLines
      ++ ["        return x"]
  where
    indent :: [Text] -> [Text]
    indent = map ("        " <>)

-- Type-level recursion ends here.
instance
  ( TranslateLayerCode b i n l,
    TranslateLayerCode b i' n' l'
  ) =>
  Translatable PyTorchGen (Network b '[Layer b i n l, Layer b i' n' l'])
  where
  type TargetStructure PyTorchGen (Network b '[Layer b i n l, Layer b i' n' l']) = ()
  translate (l0 :=> l1) = do
    translateLayer l0
    translateLayer l1

-- Type-Level recursion starts and ends here.
instance
  ( TranslateLayerCode b i n l,
    TranslateLayerCode b i' n' l',
    Translatable PyTorchGen (Network b (Layer b i'' n'' l'' ': ls)),
    ls ~ (z ': zs) -- ensures at least one more layer
  ) =>
  -- We have to explicitly match the number of layers here, otherwise the
  -- compiler does not know which instance to use.
  Translatable PyTorchGen (Network b (Layer b i n l ': Layer b i' n' l' ': Layer b i'' n'' l'' ': ls))
  where
  type TargetStructure PyTorchGen (Network b (Layer b i n l ': Layer b i' n' l' ': Layer b i'' n'' l'' ': ls)) = ()
  translate (l0 :>: l1 :>: ls) = do
    translateLayer l0
    translateLayer l1
    void $ translate ls

class TranslateLayerCode b i n l where
  translateLayer :: Layer b i n l -> PyTorchGen ()

class ShowActivation af where
  showActivation :: af -> Text -> Text

instance ShowActivation ReLU where
  showActivation _ var = T.concat [var, " = torch.relu(", var, ")"]

instance ShowActivation Softmax where
  showActivation _ var = T.concat [var, " = torch.softmax(", var, ", dim=1)"]

instance (KnownNat i, KnownNat n, ShowActivation af) => TranslateLayerCode b i n (LinearLayer i n af op) where
  translateLayer (Linear (LinearLayer _af _op _mods)) = do
    idx <- gets counter
    let layerName = T.pack $ printf "fc%d" idx
    modify $ \s ->
      s
        { layerDefs = layerDefs s ++ [T.concat ["self.", layerName, " = nn.Linear(", T.pack (show i), ", ", T.pack (show n), ")"]],
          forwardDefs =
            forwardDefs s
              ++ [ T.concat ["x = self.", layerName, "(x)"],
                   showActivation _af "x"
                 ],
          counter = idx + 1
        }
    where
      i = natVal (Proxy @i)
      n = natVal (Proxy @n)

instance
  (KnownNat i) =>
  TranslateLayerCode b i i (Drop.Drop b i)
  where
  translateLayer (Drop.DropLayer (Drop.Drop p)) = do
    idx <- gets counter
    modify $ \s ->
      s
        { layerDefs = layerDefs s ++ [T.pack $ printf "self.dropout%d = nn.Dropout(p=%f)" idx p],
          forwardDefs = forwardDefs s ++ [T.pack $ printf "x = self.dropout%d(x)" idx],
          counter = idx + 1
        }

instance
  ( Translatable PyTorchGen (Network b ls),
    ShowActivation af
  ) =>
  TranslateLayerCode b i n (Residual.Block b ls af op)
  where
  translateLayer (Residual.Residual (Residual.Block net af _op)) = do
    idx <- gets counter
    modify $ \s ->
      s
        { layerDefs = layerDefs s ++ [T.pack $ printf "self.resblock%d = ResidualBlock%d()" idx idx],
          forwardDefs =
            forwardDefs s
              ++ [ T.pack $ printf "x = self.resblock%d(x)" idx,
                   showActivation af "x"
                 ],
          counter = idx + 1
        }
    saveSubmodule idx net af

saveSubmodule ::
  (Translatable PyTorchGen (Network b ls), ShowActivation af) =>
  Int ->
  Network b ls ->
  af ->
  PyTorchGen ()
saveSubmodule idx net af = do
  oldState <- get
  put (PyTorchState [] [] [] 0)
  translate net
  PyTorchState initLines forwardLines subs _ <- get
  put oldState
  let submodule =
        T.unlines $
          [ T.pack $ printf "class ResidualBlock%d(nn.Module):" idx,
            "    def __init__(self):",
            "        super().__init__()"
          ]
            ++ indent initLines
            ++ [ "",
                 "    def forward(self, x):",
                 "        residual = x"
               ]
            ++ indent forwardLines
            ++ [ "        x = x + residual",
                 "        " <> showActivation af "x",
                 "        return x"
               ]
  modify $ \s -> s {submodules = subs ++ [submodule]}
  where
    indent = map ("        " <>)
