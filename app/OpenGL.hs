{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module OpenGL (openGLWidget) where

import Control.Lens ((&), (.~), (^.))
import Control.Loop (numLoop, numLoopState)
import Control.Monad
import Data.Default
import Data.Typeable (cast)
import Data.Vector (Vector, generate, singleton, (!))
import qualified Data.Vector as DV
import qualified Data.Vector.Storable as V
import Foreign.C.String
import Foreign.Marshal.Alloc
import Foreign.Ptr
import Foreign.Storable
import GHC.TypeLits
import Graphics.GL
import Heuron.V1.Batched.Network
  ( IteratableNetwork (forLayerIn),
    MadeOfLayers,
    Network,
  )
import Monomer
import qualified Monomer.Lens as L
import Monomer.Widgets.Single
import Types

data OpenGLWidgetMsg
  = OpenGLWidgetInit !GLuint !(Ptr GLuint) !(Ptr GLuint) !(Ptr GLuint) !(Ptr GLuint) !GLint
  deriving (Show, Eq)

data OpenGLWidgetState = OpenGLWidgetState
  { _ogsLoaded :: !Bool,
    _ogsShaderId :: !GLuint,
    _ogsLineVao :: !(Ptr GLuint),
    _ogsVao :: !(Ptr GLuint),
    _ogsVeo :: !(Ptr GLuint),
    _ogsVbo :: !(Ptr GLuint),
    _ogsBiasLoc :: !GLint,
    _ogsNetworkDescription :: !(Vector (Vector (Vector Double, Double)))
  }
  deriving (Show, Eq)

openGLWidget :: (IteratableNetwork (Network b net)) => Network b net -> WidgetNode (HeuronModel b net) e
openGLWidget net = defaultWidgetNode "openGLWidget" widget
  where
    color = red
    widget = makeOpenGLWidget color state
    state = OpenGLWidgetState False 0 nullPtr nullPtr nullPtr nullPtr 0 (networkToVectorBuffer net)

-- | networkToVectorBuffer converts a network into a matrix where each column
-- describes a layer. Each entry of said column is a neuron with its weights
-- and bias.
networkToVectorBuffer :: (IteratableNetwork (Network b net)) => Network b net -> Vector (Vector (Vector Double, Double))
networkToVectorBuffer net = forLayerIn net $ \(batchSize, inputSize, numOfNeurons) weights biases ->
  -- Creates a table which describes the structure of the network in a
  -- preformatted way. Each cell describes a neuron and its weights.
  singleton . generate numOfNeurons $ \neuronIndex ->
    let neuronWeights = weights ! neuronIndex
        neuronBias = biases ! neuronIndex
     in (neuronWeights, neuronBias)

makeOpenGLWidget :: (IteratableNetwork (Network b net)) => Color -> OpenGLWidgetState -> Widget (HeuronModel b net) e
makeOpenGLWidget color state = widget
  where
    widget =
      createSingle
        state
        def
          { singleInit = init,
            singleMerge = merge,
            singleDispose = dispose,
            singleHandleMessage = handleMessage,
            singleGetSizeReq = getSizeReq,
            singleRender = render
          }

    init wenv node = resultReqs node reqs
      where
        widgetId = node ^. L.info . L.widgetId
        path = node ^. L.info . L.path
        buffers = 2

        initOpenGL = do
          -- This needs to run in render thread
          program <- createShaderProgram
          lineVaoPtr <- malloc
          vaoPtr <- malloc
          vboPtr <- malloc
          veoPtr <- malloc
          glGenVertexArrays buffers vaoPtr
          glGenBuffers buffers vboPtr
          glGenBuffers buffers veoPtr
          biasLoc <- withCString "bias" $ \bias -> glGetUniformLocation program bias

          return $ OpenGLWidgetInit program lineVaoPtr vaoPtr veoPtr vboPtr biasLoc

        reqs = [RunInRenderThread widgetId path initOpenGL]

    merge wenv node oldNode oldState = resultNode newNode
      where
        newNet = wenv ^. L.model . heuronModelNet
        newNode =
          node
            & L.widget .~ makeOpenGLWidget color (oldState {_ogsNetworkDescription = networkToVectorBuffer newNet})

    dispose wenv node = resultReqs node reqs
      where
        OpenGLWidgetState _ shaderId lineVaoPtr vaoPtr veoPtr vboPtr biasLoc net = state
        widgetId = node ^. L.info . L.widgetId
        path = node ^. L.info . L.path
        buffers = 2

        disposeOpenGL = do
          -- This needs to run in render thread
          glDeleteProgram shaderId
          glDeleteVertexArrays buffers vaoPtr
          glDeleteBuffers buffers vboPtr
          glDeleteBuffers buffers veoPtr
          free lineVaoPtr
          free vaoPtr
          free vboPtr
          free veoPtr

        reqs = [RunInRenderThread widgetId path disposeOpenGL]

    handleMessage :: (IteratableNetwork (Network b net)) => SingleMessageHandler (HeuronModel b net) e
    handleMessage wenv node target msg = case cast msg of
      Just (OpenGLWidgetInit shaderId lineVao vao veo vbo biasLoc) -> Just result
        where
          newState = state {_ogsLoaded = True, _ogsShaderId = shaderId, _ogsLineVao = lineVao, _ogsVao = vao, _ogsVeo = veo, _ogsVbo = vbo}
          newNode =
            node
              & L.widget .~ makeOpenGLWidget color newState
          result = resultReqs newNode [RenderOnce]
      _else -> Nothing

    getSizeReq wenv node = (sizeReqW, sizeReqH)
      where
        sizeReqW = expandSize 100 1
        sizeReqH = expandSize 100 1

    render wenv node renderer =
      when (_ogsLoaded state) $
        createRawTask renderer $
          doInScissor winSize dpr offset activeVp $
            numLoop 0 (DV.length netDesc - 1) $ \layerIndex ->
              let numOfNeurons = DV.length (netDesc ! layerIndex)
                  numOfInputs = DV.length (fst $ netDesc ! layerIndex ! 0)
               in numLoop 0 (numOfNeurons - 1) $ \neuronIndex -> do
                    let neuronYOffset = fromIntegral neuronIndex - fromIntegral numOfNeurons / 2
                        (myX, myY) = (100 + fromIntegral layerIndex * layerWidth + rx, neuronYOffset * neuronHeight + ry + rh / 2)
                        octagon = octagonAt (myX, myY) 4
                        biasLoc = _ogsBiasLoc state
                    glUniform1f biasLoc (realToFrac $ snd $ netDesc ! layerIndex ! neuronIndex)
                    drawVertices state (toVectorVAO winSize offset color octagon) (toVectorEAO octagonElements)
      where
        -- Draw lines for weights:
        -- TODO

        dpr = wenv ^. L.dpr
        winSize = wenv ^. L.windowSize
        activeVp = wenv ^. L.viewport
        offset = wenv ^. L.offset

        style = currentStyle wenv node
        nodeVp = getContentArea node style

        layerWidth = 100
        neuronHeight = 10

        Rect rx ry rw rh = nodeVp
        netDesc = _ogsNetworkDescription state

octagonAt :: (Double, Double) -> Double -> [(Double, Double)]
octagonAt (x, y) radius = origin : corners
  where
    origin = (x, y)
    corners = take 8 $ [(x + radius * cos a, y + radius * sin a) | a <- [0, pi / 4 ..]]

octagonElements :: [GLuint]
octagonElements =
  [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 6, 0, 6, 7, 0, 7, 8, 0, 8, 1]

doInScissor :: Size -> Double -> Point -> Rect -> IO () -> IO ()
doInScissor winSize dpr offset vp action = do
  glEnable GL_SCISSOR_TEST
  -- OpenGL's Y axis increases from bottom to top
  glScissor (round (rx + ox)) (round $ winH - ry - oy - rh) (round rw) (round rh)
  action
  glDisable GL_SCISSOR_TEST
  where
    winH = winSize ^. L.h * dpr
    Point ox oy = mulPoint dpr offset
    Rect rx ry rw rh = mulRect dpr vp

toVectorVAO :: Size -> Point -> Color -> [(Double, Double)] -> V.Vector Float
toVectorVAO (Size w h) (Point ox oy) (Color r g b a) points = vec
  where
    px x = realToFrac $ (x + ox - w / 2) / (w / 2)
    -- OpenGL's Y axis increases from bottom to top
    py y = realToFrac $ (h / 2 - y - oy) / (h / 2)
    col c = realToFrac (fromIntegral c / 255)
    row (x, y) = [px x, py y, 0, col r, col g, col b]
    vec = V.fromList . concat $ row <$> points

toVectorEAO :: [GLuint] -> V.Vector GLuint
toVectorEAO = V.fromList

drawVertices ::
  forall a c.
  (Storable a, Storable c) =>
  OpenGLWidgetState ->
  V.Vector a ->
  V.Vector c ->
  IO ()
drawVertices state vertices elements = do
  lineVao <- peek lineVaoPtr
  vao <- peek vaoPtr
  veo <- peek veoPtr
  vbo <- peek vboPtr

  glBindVertexArray vao
  glBindBuffer GL_ELEMENT_ARRAY_BUFFER veo
  glBindBuffer GL_ARRAY_BUFFER vbo

  -- Copies raw data from vector to OpenGL memory
  V.unsafeWith vertices $ \vertsPtr ->
    glBufferData
      GL_ARRAY_BUFFER
      (fromIntegral (V.length vertices * floatSize))
      (castPtr vertsPtr)
      GL_STATIC_DRAW

  V.unsafeWith elements $ \elemsPtr ->
    glBufferData
      GL_ELEMENT_ARRAY_BUFFER
      (fromIntegral (V.length elements * uintSize))
      (castPtr elemsPtr)
      GL_STATIC_DRAW

  -- The vertex shader expects two arguments. Position:
  glVertexAttribPointer 0 3 GL_FLOAT GL_FALSE (fromIntegral (floatSize * 6)) nullPtr
  glEnableVertexAttribArray 0

  -- Color:
  glVertexAttribPointer 1 3 GL_FLOAT GL_FALSE (fromIntegral (floatSize * 6)) (nullPtr `plusPtr` (floatSize * 3))
  glEnableVertexAttribArray 1

  glUseProgram shaderId
  glBindVertexArray vao
  glDrawElements GL_TRIANGLES (fromIntegral $ V.length elements) GL_UNSIGNED_INT nullPtr
  where
    floatSize = sizeOf (undefined :: Float)
    uintSize = sizeOf (undefined :: GLuint)
    OpenGLWidgetState _ shaderId lineVaoPtr vaoPtr veoPtr vboPtr biasLoc net = state

createShaderProgram :: IO GLuint
createShaderProgram = do
  shaderProgram <- glCreateProgram
  vertexShader <- compileShader GL_VERTEX_SHADER "resources/vert.glsl"
  fragmentShader <- compileShader GL_FRAGMENT_SHADER "resources/frag.glsl"

  glAttachShader shaderProgram vertexShader
  glAttachShader shaderProgram fragmentShader

  glLinkProgram shaderProgram
  checkProgramLink shaderProgram

  glDeleteShader vertexShader
  glDeleteShader fragmentShader

  return shaderProgram

compileShader :: GLenum -> FilePath -> IO GLuint
compileShader shaderType shaderFile = do
  shader <- glCreateShader shaderType
  shaderSource <- readFile shaderFile >>= newCString

  alloca $ \shadersStr -> do
    shadersStr `poke` shaderSource
    glShaderSource shader 1 shadersStr nullPtr
    glCompileShader shader
    checkShaderCompile shader

  return shader

checkProgramLink :: GLuint -> IO ()
checkProgramLink programId = do
  alloca $ \successPtr -> do
    alloca $ \infoLogPtr -> do
      glGetProgramiv programId GL_LINK_STATUS successPtr
      success <- peek successPtr

      when (success <= 0) $ do
        glGetProgramInfoLog programId 512 nullPtr infoLogPtr
        putStrLn =<< peekCString infoLogPtr

checkShaderCompile :: GLuint -> IO ()
checkShaderCompile shaderId = do
  alloca $ \successPtr ->
    alloca $ \infoLogPtr -> do
      glGetShaderiv shaderId GL_COMPILE_STATUS successPtr
      success <- peek successPtr

      when (success <= 0) $ do
        glGetShaderInfoLog shaderId 512 nullPtr infoLogPtr
        putStrLn "Failed to compile shader "
