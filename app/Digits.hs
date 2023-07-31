{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE LambdaCase #-}

module Digits where

import Control.Monad (unless, when)
import Control.Monad.Except (ExceptT, MonadError (..), runExceptT)
import Data.Bits
import Data.ByteString as BS
import qualified Data.ByteString as BS
import Data.Data (Typeable)
import Data.Functor ((<&>))
import qualified Data.Vector as V
import Data.Word
import Heuron.V1.Batched
import Linear.V
import Streaming (MonadIO, liftIO)
import Streaming.ByteString (ByteStream)
import qualified Streaming.ByteString as Q
import qualified Streaming.ByteString.Char8 as Q8
import Streaming.Prelude (each, next, yield)
import qualified Streaming.Prelude as S
import System.IO
import qualified System.IO as IO

data StreamingFile m a = StreamingFile
  { sfHandle :: !Handle,
    sfStream :: !(ByteStream m a)
  }

streamMNISTImages :: FilePath -> IO (S.Stream (S.Of (V.Vector Word8)) (ExceptT HeuronError IO) ())
streamMNISTImages fp = do
  h <- openFile fp ReadMode
  (magic, raw) <- parseWord32 $ Q.fromHandle h
  when (magic /= 2051) $ error "invalid magic number"
  (numOfItems, raw) <- parseWord32 raw
  print $ "Number of Image items: " ++ show numOfItems
  (numOfRows, raw) <- parseWord32 raw
  print $ "Number of Image rows: " ++ show numOfRows
  (numOfColumns, raw) <- parseWord32 raw
  print $ "Number of Image columns: " ++ show numOfColumns
  return $ go (fromIntegral numOfRows) (fromIntegral numOfColumns) (StreamingFile h raw)
  where
    go :: Int -> Int -> StreamingFile (ExceptT HeuronError IO) a -> S.Stream (S.Of (V.Vector Word8)) (ExceptT HeuronError IO) ()
    go numOfRows numOfColumns s@(StreamingFile h raw) = do
      eof <- liftIO $ IO.hIsEOF h
      unless eof $ do
        (img, s') <-
          liftIO (nextImage numOfRows numOfColumns s) >>= \case
            Left err -> error $ show err
            Right (img, s') -> return (img, s')
        yield img
        go numOfRows numOfColumns s'

    nextImage :: Int -> Int -> StreamingFile (ExceptT HeuronError IO) a -> IO (Either HeuronError (V.Vector Word8, StreamingFile (ExceptT HeuronError IO) a))
    nextImage numOfRows numOfColumns (StreamingFile h raw) = do
      (imgBytes, raw) <-
        runExceptT (nextBytesN (numOfRows * numOfColumns) raw) >>= \case
          Right (Right (bytes, raw)) -> return (bytes, raw)
          Left err -> error $ show err
          _else -> error "unexpected"
      let numOfPixels = fromIntegral numOfRows * fromIntegral numOfColumns
          img = V.generate numOfPixels $ \i -> fromIntegral $ BS.index imgBytes i
      return $ Right (img, StreamingFile h raw)

streamMNISTLabels :: FilePath -> IO (S.Stream (S.Of (V.Vector Double)) (ExceptT HeuronError IO) ())
streamMNISTLabels fp = do
  h <- openFile fp ReadMode
  (magic, raw) <- parseWord32 $ Q.fromHandle h
  when (magic /= 2049) $ error "invalid magic number"
  (numOfItems, raw) <- parseWord32 raw
  print $ "Number of Label items: " ++ show numOfItems
  return $ go (fromIntegral numOfItems) (StreamingFile h raw)
  where
    go :: Int -> StreamingFile (ExceptT HeuronError IO) a -> S.Stream (S.Of (V.Vector Double)) (ExceptT HeuronError IO) ()
    go numOfItems s@(StreamingFile h raw) = do
      eof <- liftIO $ IO.hIsEOF h
      unless eof $ do
        (label, s') <-
          liftIO (nextLabel s) >>= \case
            Right (b, raw) -> return (b, raw)
            Left err -> error $ show err
        yield label
        go numOfItems s'

    nextLabel :: StreamingFile (ExceptT HeuronError IO) a -> IO (Either HeuronError (V.Vector Double, StreamingFile (ExceptT HeuronError IO) a))
    nextLabel (StreamingFile h raw) = do
      (label, raw) <-
        runExceptT (nextByte raw) >>= \case
          Right (Right (b, raw)) -> return (b, raw)
          Left err -> error $ show err
          _else -> error "unexpected"
      -- One-Hot encoded truth vector.
      return $ Right (V.generate 10 (\i -> if i == fromIntegral label then 1.0 else 0.0), StreamingFile h raw)

parseWord32 ::
  (Monad m) =>
  Q.ByteStream (ExceptT HeuronError m) r ->
  m (Word32, Q.ByteStream (ExceptT HeuronError m) r)
parseWord32 bs = do
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
