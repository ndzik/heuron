{-# LANGUAGE AllowAmbiguousTypes #-}
{-# LANGUAGE DataKinds #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE PolyKinds #-}
{-# LANGUAGE RankNTypes #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}
{-# LANGUAGE TypeFamilies #-}
{-# LANGUAGE UndecidableInstances #-}

module Main where

import Codec.Serialise
import Control.Lens
import Control.Monad.Except (ExceptT, runExceptT)
import qualified Data.ByteString.Lazy as BSL
import Data.Data (Proxy (..))
import Data.Functor ((<&>))
import Digits
import GHC.TypeLits (natVal)
import GHC.TypeNats (KnownNat)
import Heuron.Functions
import Heuron.V1
import Heuron.V1.Batched
import Heuron.V1.Batched.Layer
import Linear.V
import Monomer
import Streaming (liftIO)
import qualified Streaming as S
import qualified Streaming.Prelude as S
import System.Random (getStdGen, mkStdGen)
import System.Random.Stateful (StateGenM (StateGenM), globalStdGen, newIOGenM)
import Text.Printf (printf)
import Types
import View

pathToMNISTLabel :: FilePath
pathToMNISTLabel = "./data/train-labels-idx1-ubyte"

pathToMNISTImage :: FilePath
pathToMNISTImage = "./data/train-images-idx3-ubyte"

scaledBy :: (KnownNat n, KnownNat m) => Double -> Input n m Double -> Input n m Double
scaledBy x = fmap (fmap (* x))

main :: forall pixelCount batchSize numOfImages hiddenNeuronCount. (hiddenNeuronCount ~ 32, pixelCount ~ 784, batchSize ~ 100, numOfImages ~ 57321) => IO ()
main = do
  rng <- newIOGenM (mkStdGen 42069)

  -- Describe network.
  let learningRate = 0.25
  inputLayer <- mkLayer $ do
    inputs @pixelCount
    neuronsWith @hiddenNeuronCount rng $ weightsScaledBy (1 / 784)
    activationF ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  [hiddenLayer00, hiddenLayer01] <- mkLayers 2 $ do
    neuronsWith @hiddenNeuronCount rng $ weightsScaledBy (1 / 32)
    activationF ReLU
    optimizerFunction (StochasticGradientDescent learningRate)

  outputLayer <- mkLayer $ do
    neurons @10 rng
    activationF Softmax
    optimizerFunction (StochasticGradientDescent learningRate)
  let ann = inputLayer :>: hiddenLayer00 :>: hiddenLayer01 =| outputLayer
      initialTrainerState = TrainerState ann CategoricalCrossEntropy

  let producer env sendMsg = do
        -- Train network.
        labels <- streamMNISTLabels pathToMNISTLabel
        imgs <- streamMNISTImages @pixelCount pathToMNISTImage
        let s = streamOfSize @batchSize $ S.zip labels imgs
            trainNetwork (ts, epoch) (labels, images) = do
              (tr, ts') <- liftIO $ runTrainer (oneEpoch images labels) ts
              liftIO . sendMsg . HeuronUpdate $ UpdateEvent (tr ^. trainingResultLoss) (tr ^. trainingResultAccuracy) epoch (ts' ^. network . to viewNetFromHeuronNet)
              return (ts', epoch + 1)

        print "Starting training..."
        _res <- runExceptT $ S.foldM_ trainNetwork (pure (initialTrainerState, 0)) pure s
        print "Done training."

  let config =
        [ appWindowTitle "Heuron",
          appTheme darkTheme,
          appScaleFactor 1.5,
          appFontDef
            "Regular"
            "./resources/Hasklig-Regular.otf",
          appInitEvent HeuronInit
        ]
      model =
        HeuronModel
          { _heuronModelNet = viewNetFromHeuronNet ann,
            _heuronModelAvgLoss = 0.00,
            _heuronModelAccuracy = 0.00,
            _heuronModelCurrentEpoch = 0,
            _heuronModelMaxEpochs = maxEpochs
          }
  startApp model (handleEvent producer) buildUI config
  where
    maxEpochs = natVal (Proxy @numOfImages) `div` natVal (Proxy @batchSize)

handleEvent ::
  (WidgetEnv HeuronModel HeuronEvent -> ProducerHandler HeuronEvent) ->
  WidgetEnv HeuronModel HeuronEvent ->
  WidgetNode HeuronModel HeuronEvent ->
  HeuronModel ->
  HeuronEvent ->
  [AppEventResponse HeuronModel HeuronEvent]
handleEvent producer we node model evt = case evt of
  HeuronInit -> [Producer $ producer we]
  HeuronUpdate (UpdateEvent avgLoss acc currentEpoch newNet) ->
    [ Model
        model
          { _heuronModelAvgLoss = avgLoss,
            _heuronModelAccuracy = acc,
            _heuronModelCurrentEpoch = currentEpoch,
            _heuronModelNet = newNet
          }
    ]

streamOfSizeFiltered ::
  forall b.
  (KnownNat b) =>
  ((V 10 Double, V 784 Double) -> Bool) ->
  S.Stream (S.Of (V 10 Double, V 784 Double)) (ExceptT HeuronError IO) () ->
  S.Stream (S.Of (Input b 10 Double, Input b 784 Double)) (ExceptT HeuronError IO) ()
streamOfSizeFiltered f s =
  let batchSize = fromIntegral $ natVal (Proxy @b)
      substreams = S.chunksOf batchSize . S.filter f $ s
   in S.mapsM collapseSubstreams substreams
  where
    collapseSubstreams ::
      S.Stream
        (S.Of (V 10 Double, V 784 Double))
        (ExceptT HeuronError IO)
        x ->
      ExceptT HeuronError IO (S.Of (V b (V 10 Double), V b (V 784 Double)) x)
    collapseSubstreams substream = do
      labelsAndImages <- S.toList substream
      let collapsed = S.mapOf (S.bimap (mkV'' @b) (mkV'' @b) . unzip) labelsAndImages
      return collapsed
    mkV'' :: forall b a. (KnownNat b) => [a] -> V b a
    mkV'' as = case mkV @b as of
      Nothing -> error "mkV''"
      Just v -> v

streamOfSize ::
  forall b.
  (KnownNat b) =>
  S.Stream (S.Of (V 10 Double, V 784 Double)) (ExceptT HeuronError IO) () ->
  S.Stream (S.Of (Input b 10 Double, Input b 784 Double)) (ExceptT HeuronError IO) ()
streamOfSize = streamOfSizeFiltered @b (const True)
