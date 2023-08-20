{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE RankNTypes #-}

module OpenGL (openGLWidget) where

import Control.Lens ((&), (.~), (^.))
import Control.Loop (numLoop, numLoopFold)
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
  = OpenGLWidgetInit !GLuint !Int !Int !(Ptr GLuint) !(Ptr GLuint) !(Ptr GLuint) !(Ptr GLuint) !(Ptr GLuint) !(Ptr GLuint) !GLint
  deriving (Show, Eq)

data OpenGLWidgetState = OpenGLWidgetState
  { _ogsLoaded :: !Bool,
    _ogsNetworkLoaded :: !Bool,
    _ogsShaderId :: !GLuint,
    _ogsLineVao :: !(Ptr GLuint),
    _ogsLineVeo :: !(Ptr GLuint),
    _ogsLineVbo :: !(Ptr GLuint),
    _ogsVao :: !(Ptr GLuint),
    _ogsVeo :: !(Ptr GLuint),
    _ogsVbo :: !(Ptr GLuint),
    _ogsBiasLoc :: !GLint,
    _ogsNumOfNeurons :: !Int,
    _ogsNumOfLines :: !Int
  }
  deriving (Show, Eq)

openGLWidget :: ViewNetwork -> WidgetNode HeuronModel e
openGLWidget net = defaultWidgetNode "openGLWidget" widget
  where
    color = red
    widget = makeOpenGLWidget net color state
    state = OpenGLWidgetState False False 0 nullPtr nullPtr nullPtr nullPtr nullPtr nullPtr 0 0 0

makeOpenGLWidget :: ViewNetwork -> Color -> OpenGLWidgetState -> Widget HeuronModel e
makeOpenGLWidget initNet color state = widget
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

          let (_, _, numOfNeurons, _, _, numOfLines) = generateNetworkVectors wenv node initNet

          -- Load initial network nodes.
          vaoPtr <- malloc
          vboPtr <- malloc
          veoPtr <- malloc
          glGenVertexArrays buffers vaoPtr
          glGenBuffers buffers vboPtr
          glGenBuffers buffers veoPtr
          biasLoc <- withCString "bias" $ \bias -> glGetUniformLocation program bias

          -- Load initial network lines.
          lineVaoPtr <- malloc
          lineVboPtr <- malloc
          lineVeoPtr <- malloc
          glGenVertexArrays buffers lineVaoPtr
          glGenBuffers buffers lineVboPtr
          glGenBuffers buffers lineVeoPtr

          return $ OpenGLWidgetInit program numOfNeurons numOfLines lineVaoPtr lineVeoPtr lineVboPtr vaoPtr veoPtr vboPtr biasLoc

        reqs = [RunInRenderThread widgetId path initOpenGL]

    merge wenv node oldNode oldState = if _ogsNetworkLoaded oldState then resultNode newNode else resultReqs newNode reqs
      where
        reloadVertices = do
          let (neuronVertices, neuronElements, _, lineVertices, lineElements, _) = generateNetworkVectors wenv newNode initNet
          loadVertices (vaoPtr, vboPtr, veoPtr) neuronVertices neuronElements
          loadVertices (lineVaoPtr, lineVboPtr, lineVeoPtr) lineVertices lineElements
        OpenGLWidgetState _ _ _ lineVaoPtr lineVeoPtr lineVboPtr vaoPtr veoPtr vboPtr _ _ _ = oldState
        widgetId = node ^. L.info . L.widgetId
        path = node ^. L.info . L.path
        reqs = [RunInRenderThread widgetId path reloadVertices]
        newNet = wenv ^. L.model . heuronModelNet
        newNode =
          node
            & L.widget .~ makeOpenGLWidget newNet color oldState {_ogsNetworkLoaded = True}

    dispose wenv node = resultReqs node reqs
      where
        OpenGLWidgetState _ _ shaderId lineVaoPtr lineVeoPtr lineVboPtr vaoPtr veoPtr vboPtr biasLoc numOfNeurons numOfLines = state
        widgetId = node ^. L.info . L.widgetId
        path = node ^. L.info . L.path
        buffers = 2

        disposeOpenGL = do
          -- This needs to run in render thread
          glDeleteProgram shaderId
          forM_ [vaoPtr, lineVaoPtr] $ \ptr -> do
            glDeleteVertexArrays buffers ptr
            free ptr
          forM_ [vboPtr, veoPtr, lineVboPtr, lineVeoPtr] $ \ptr -> do
            glDeleteBuffers buffers ptr
            free ptr

        reqs = [RunInRenderThread widgetId path disposeOpenGL]

    handleMessage :: SingleMessageHandler HeuronModel e
    handleMessage wenv node target msg = case cast msg of
      Just (OpenGLWidgetInit shaderId numOfNeurons numOfLines lineVao lineVeo lineVbo vao veo vbo biasLoc) -> Just result
        where
          newState = state {_ogsLoaded = True, _ogsShaderId = shaderId, _ogsLineVao = lineVao, _ogsLineVeo = lineVeo, _ogsLineVbo = lineVbo, _ogsVao = vao, _ogsVeo = veo, _ogsVbo = vbo, _ogsNumOfNeurons = numOfNeurons, _ogsNumOfLines = numOfLines}
          newNode =
            node
              & L.widget .~ makeOpenGLWidget initNet color newState
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
            drawNetwork state initNet
      where
        dpr = wenv ^. L.dpr
        winSize = wenv ^. L.windowSize
        activeVp = wenv ^. L.viewport
        offset = wenv ^. L.offset

        style = currentStyle wenv node
        nodeVp = getContentArea node style

        Rect rx ry rw rh = nodeVp
        layeredWeights = initNet ^. viewNetworkWeights
        layeredBiases = initNet ^. viewNetworkBiases

generateNetworkVectors :: WidgetEnv HeuronModel e -> WidgetNode HeuronModel e -> ViewNetwork -> (V.Vector Float, V.Vector GLuint, Int, V.Vector Float, V.Vector GLuint, Int)
generateNetworkVectors wenv node initNet =
  let networkOffset = 300
      layerWidth = 100
      winSize = wenv ^. L.windowSize
      offset = wenv ^. L.offset
      (neuronVertices, numOfNeurons, lineVertices, numOfLines) = numLoopFold 0 (DV.length layeredWeights - 1) (mempty, 0, mempty, 0) $ \(nsV, numNs, linesV, lineNs) layerIndex ->
        let numOfNeurons = DV.length (layeredWeights ! layerIndex)
            numOfInputs =
              if layerIndex == 0
                then -- Ignore inputs to network, to many lines...
                  1
                else DV.length (layeredWeights ! layerIndex ! 0)
            (neuronVerticesI, linesToNeuronI) = numLoopFold 0 (numOfNeurons - 1) (mempty, mempty) $ \(vs, ls) neuronIndex ->
              let neuronYOffset = fromIntegral neuronIndex - fromIntegral numOfNeurons / 2
                  (myX, myY) = (networkOffset + fromIntegral layerIndex * layerWidth + rx, neuronYOffset * neuronHeight + ry + rh / 2)
                  octagon = octagonAt (myX, myY) octagonRadius
                  lineVerticesI = numLoopFold 0 (numOfInputs - 1) mempty $ \lsI inputIndex ->
                    let inputYOffset = fromIntegral inputIndex - fromIntegral numOfInputs / 2
                        (inputX, inputY) = (networkOffset + fromIntegral (layerIndex - 1) * layerWidth + rx, inputYOffset * neuronHeight + ry + rh / 2)
                     in lsI <> toVectorVAO winSize offset blue [(inputX+octagonRadius, inputY), (myX-octagonRadius, myY)]
               in (vs <> toVectorVAO winSize offset red octagon, ls <> lineVerticesI)
         in (nsV <> neuronVerticesI, numNs + numOfNeurons, linesV <> linesToNeuronI, lineNs + numOfNeurons * numOfInputs)
      neuronElements = numLoopFold 0 (numOfNeurons - 1) mempty $ \es idx -> es <> V.map (+ (fromIntegral $ idx * 9)) (toVectorEAO octagonElements)
      lineElements = numLoopFold 0 (numOfLines - 1) mempty $ \es idx -> es <> V.map (+ (fromIntegral $ idx * 2)) (toVectorEAO [0, 1])
   in (neuronVertices, neuronElements, V.length neuronElements, lineVertices, lineElements, V.length lineElements)
  where
    layeredWeights = initNet ^. viewNetworkWeights
    octagonRadius = 6
    neuronHeight = octagonRadius * 2 + 2
    style = currentStyle wenv node
    nodeVp = getContentArea node style
    Rect rx ry rw rh = nodeVp

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

loadVertices ::
  forall a c.
  (Storable a, Storable c) =>
  (Ptr GLuint, Ptr GLuint, Ptr GLuint) ->
  V.Vector a ->
  V.Vector c ->
  IO ()
loadVertices (ptrVao, ptrVbo, ptrVeo) vertices elements = do
  peek ptrVao >>= glBindVertexArray
  peek ptrVbo >>= glBindBuffer GL_ARRAY_BUFFER

  -- Copies raw data from vector to OpenGL memory
  V.unsafeWith vertices $ \vertsPtr ->
    glBufferData
      GL_ARRAY_BUFFER
      (fromIntegral (V.length vertices * floatSize))
      (castPtr vertsPtr)
      GL_STATIC_DRAW

  -- The vertex shader expects two arguments. Position:
  glVertexAttribPointer 0 3 GL_FLOAT GL_FALSE (fromIntegral (floatSize * 6)) nullPtr
  glEnableVertexAttribArray 0

  -- Color:
  glVertexAttribPointer 1 3 GL_FLOAT GL_FALSE (fromIntegral (floatSize * 6)) (nullPtr `plusPtr` (floatSize * 3))
  glEnableVertexAttribArray 1

  peek ptrVeo >>= glBindBuffer GL_ELEMENT_ARRAY_BUFFER
  V.unsafeWith elements $ \elemsPtr ->
    glBufferData
      GL_ELEMENT_ARRAY_BUFFER
      (fromIntegral (V.length elements * uintSize))
      (castPtr elemsPtr)
      GL_STATIC_DRAW
  where
    floatSize = sizeOf (undefined :: Float)
    uintSize = sizeOf (undefined :: GLuint)

drawVertices ::
  forall a c.
  (Storable a, Storable c) =>
  OpenGLWidgetState ->
  V.Vector a ->
  V.Vector c ->
  IO ()
drawVertices state vertices elements = do
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
  glDrawElements GL_TRIANGLES (fromIntegral $ V.length elements) GL_UNSIGNED_INT nullPtr
  where
    floatSize = sizeOf (undefined :: Float)
    uintSize = sizeOf (undefined :: GLuint)
    OpenGLWidgetState _ _ shaderId lineVaoPtr lineVeoPtr lineVboPtr vaoPtr veoPtr vboPtr biasLoc numOfNeurons numOfLines = state

drawNetwork :: OpenGLWidgetState -> ViewNetwork -> IO ()
drawNetwork state net = do
  let layeredWeights = net ^. viewNetworkWeights
      layeredBiases = net ^. viewNetworkBiases

  void $ numLoopFold 0 (DV.length layeredWeights - 1) (return 0) $ \ni layerIndex -> do
    n <- ni
    let numOfNeuronsInLayer = DV.length (layeredWeights DV.! layerIndex)
        numOfInputs = DV.length (layeredWeights DV.! layerIndex ! 0)
    numLoop 0 (numOfNeuronsInLayer - 1) $ \neuronIndex -> do
      -- Draw neuron edges.
      peek lineVaoPtr >>= glBindVertexArray
      peek lineVboPtr >>= glBindBuffer GL_ARRAY_BUFFER
      peek lineVeoPtr >>= glBindBuffer GL_ELEMENT_ARRAY_BUFFER
      glUseProgram shaderId
      numLoop 0 (numOfInputs - 1) $ \inputIndex -> do
        let weight = layeredWeights ! layerIndex ! neuronIndex ! inputIndex
            numOfElements = 2
        -- Some random scaling to make the lines visible and better show
        -- progress of learning.
        if weight <= 0.01 then
          glUniform1f colorLoc (realToFrac $ 100 * weight)
        else
          glUniform1f colorLoc (realToFrac $ weight)
        glDrawElements GL_LINES numOfElements GL_UNSIGNED_INT (nullPtr `plusPtr` (uintSize * fromIntegral numOfElements * (n + numOfNeuronsInLayer + neuronIndex * numOfInputs + inputIndex)))

      -- Draw neuron node.
      peek vaoPtr >>= glBindVertexArray
      peek vboPtr >>= glBindBuffer GL_ARRAY_BUFFER
      peek veoPtr >>= glBindBuffer GL_ELEMENT_ARRAY_BUFFER
      glUseProgram shaderId
      let bias = layeredBiases ! layerIndex ! neuronIndex
          numOfElements = 24
      glUniform1f colorLoc (realToFrac bias)
      glDrawElements GL_TRIANGLES numOfElements GL_UNSIGNED_INT (nullPtr `plusPtr` (uintSize * fromIntegral numOfElements * (n + neuronIndex)))
    return (n + numOfNeuronsInLayer)
  where
    floatSize = sizeOf (undefined :: Float)
    uintSize = sizeOf (undefined :: GLuint)
    OpenGLWidgetState _ _ shaderId lineVaoPtr lineVeoPtr lineVboPtr vaoPtr veoPtr vboPtr colorLoc numOfNeurons numOfLines = state

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
