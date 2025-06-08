{-# LANGUAGE StandaloneKindSignatures #-}
{-# LANGUAGE UndecidableInstances #-}

module Heuron.V2.Residual where

import Control.Lens
import Control.Monad.State
import Data.Default
import Data.Kind (Constraint)
import Data.Maybe
import GHC.TypeLits
import Heuron.V2.Layer
import qualified Heuron.V2.Network as Network

data ResidualBlock b ls af op where
  ResidualBlock :: Network.Network b ls -> af -> op -> ResidualBlock b ls af op

newtype instance Layer (b :: Nat) (i :: Nat) (n :: Nat) (ResidualBlock b ls af op) = Residual (ResidualBlock b ls af op)

data BlockBuilderState (i :: Nat) (n :: Nat) af op = BlockBuilderState
  { _blockAf :: !(Maybe af),
    _blockOp :: !(Maybe op)
  }

instance Default (BlockBuilderState i n af op) where
  def = BlockBuilderState Nothing Nothing

type BlockT (b :: Nat) (i :: Nat) (n :: Nat) af op m a = StateT (BlockBuilderState i n af op) m a

type CheckCompatibleOutput :: * -> * -> Constraint
type family CheckCompatibleOutput l ls where
  CheckCompatibleOutput (Layer b i n l) (Layer b i' n' l') = (n ~ n')

type CheckCompatibleInput :: * -> * -> Constraint
type family CheckCompatibleInput l ls where
  CheckCompatibleInput (Layer b i n l) (Layer b i' n' l') = (i ~ i')

makeLenses ''BlockBuilderState

mkBlock ::
  forall b i n m af op ls.
  ( KnownNat b,
    KnownNat i,
    KnownNat n,
    Monad m,
    Network.CheckValidLayers (Layer b i n (ResidualBlock b ls af op)) (Network.InputLayerOfNetwork ls),
    CheckCompatibleInput (Layer b i n (ResidualBlock b ls af op)) (Network.InputLayerOfNetwork ls),
    CheckCompatibleOutput (Layer b i n (ResidualBlock b ls af op)) (Network.OutputLayerOfNetwork ls)
  ) =>
  BlockT b i n af op m (Network.Network b ls) ->
  m (Layer b i n (ResidualBlock b ls af op))
mkBlock builder = evalStateT (builder >>= initialize) def
  where
    initialize net = do
      af <- use blockAf >>= maybe (error "no activation function set") pure
      op <- use blockOp >>= maybe (error "no optimizer set") pure
      return $ Residual $ ResidualBlock net af op

inputs :: forall i n b af op m. (KnownNat i, Monad m) => BlockT b i n af op m ()
inputs = return ()

activationFunction :: (Monad m) => af -> BlockT b i n af op m ()
activationFunction af = modify $ \s -> s {_blockAf = Just af}

optimizerFunction :: (Monad m) => op -> BlockT b i n af op m ()
optimizerFunction op = modify $ \s -> s {_blockOp = Just op}
