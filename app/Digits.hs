{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}

module Digits where

import Control.Monad (when)
import Control.Monad.Except (MonadError (..), runExceptT)
import Data.Bits
import Data.ByteString as BS
import qualified Data.ByteString as BS
import Data.Data (Typeable)
import Data.Functor ((<&>))
import Data.Word
import Heuron.V1.Batched
import Streaming.ByteString (ByteStream)
import qualified Streaming.ByteString as Q
import qualified Streaming.ByteString.Char8 as Q8
import Streaming.Prelude (each, next, yield)
import qualified Streaming.Prelude as S
import System.IO

streamMNISTLabels :: FilePath -> IO ()
streamMNISTLabels fp = withFile fp ReadMode $ \h -> do
  (magic, raw) <- parseMagic $ Q.fromHandle h
  when (magic /= 2049) $ error "invalid magic number"
  (numOfItems, raw) <- parseNumOfItems raw
  print $ "Number of Label items: " ++ show numOfItems
  where
    -- TODO: Convert this to a stream. Do not use `withFile` here, we need to
    -- keep the handle open as long as the stream is alive.

    parseMagic bs = do
      runExceptT (getWord32 bs) >>= \case
        Right (Right res) -> return res
        Left err -> error $ show err
        _else -> error "unexpected"
    parseNumOfItems bs = do
      runExceptT (getWord32 bs) >>= \case
        Right (Right res) -> return res
        Left err -> error $ show err
        _else -> error "unexpected"

getWord32 :: (Monad m, (MonadError HeuronError m)) => ByteStream m r -> m (Either HeuronError (Word32, ByteStream m r))
getWord32 bs = do
  (bytes, bs) <-
    nextBytesN 4 bs >>= \case
      Left err -> throwError err
      Right (bytes, bs) -> return (bytes, bs)
  return $ Right (byteStringToWord32 bytes, bs)

byteStringToWord32 :: ByteString -> Word32
byteStringToWord32 bs = shiftL (fromIntegral n0) 24 .|. shiftL (fromIntegral n1) 16 .|. shiftL (fromIntegral n2) 8 .|. fromIntegral n3
  where
    [n0, n1, n2, n3] = BS.unpack bs

nextBytesN :: (Monad m) => Int -> ByteStream m r -> m (Either HeuronError (ByteString, ByteStream m r))
nextBytesN n bs = go n bs BS.empty
  where
    go 0 bs acc = return $ Right (acc, bs)
    go n bs acc = do
      nextByte bs >>= \case
        Left err -> return $ Left err
        Right (b, bs) -> go (n - 1) bs (BS.snoc acc b)

nextByte :: (Monad m) => ByteStream m r -> m (Either HeuronError (Word8, ByteStream m r))
nextByte s =
  Q.uncons s >>= \case
    Left res -> return . Left $ InvalidDataFile "unexpected end of file"
    Right (b, rs) -> return . Right $ (b, rs)

newtype HeuronError = InvalidDataFile String deriving (Show)
