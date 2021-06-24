from os import sep
import bpy
import zmq
import numpy as np
from mathutils import Vector
import bmesh

def get_cam_frustum(cam):
    # get vectors which define view frustum of camera
    frame = cam.data.view_frame(scene=bpy.context.scene)
    topRight = frame[0]
    bottomRight = frame[1]
    bottomLeft = frame[2]
    topLeft = frame[3]
    return topRight,bottomRight,bottomLeft,topLeft

def fill_array_with_none(xRange, yRange):
    values = np.empty((xRange.size, yRange.size), dtype=object)

    # indices for array mapping
    indexX = 0
    indexY = 0

    # filling array with None
    for x in xRange:
        for y in yRange:
            values[indexX,indexY] = (None, None)
            indexY += 1
        indexX += 1
        indexY = 0
    return values

def create_mesh(ind_vertices):
    # create new mesh
    # source: https://devtalk.blender.org/t/alternative-in-2-80-to-create-meshes-from-python-using-the-tessfaces-api/7445/3
    mesh = bpy.data.meshes.new(name='created mesh')
    facemesh = bpy.data.meshes.new(name='created face mesh')
    bm = bmesh.new()
    fm = bmesh.new()
    vertex1 = None
    vertex2 = None
    vertex3 = None
    flag = False
    #f1 = bm.faces.new( [bm.verts[0], bm.verts[1], bm.verts[2]])
    # iterate over all possible hits
    screenSpacePoints = []
    #    print("Going into for", ind_vertices.shape)
    for index, location in np.ndenumerate(ind_vertices):
        # no hit at this position
        if location is None or location[0] is None:
            continue
        # add new vertex
        screenSpacePoints.append(index)
        bm.verts.new((location[0], location[1], location[2]))
        bm.verts.ensure_lookup_table()
    # make the bmesh the object's mesh
    bm.to_mesh(mesh)  
    bm.free()  # always do this when finished

    # We're done setting up the mesh values, update mesh object and 
    # let Blender do some checks on it
    mesh.update()
    mesh.validate()

    return mesh, screenSpacePoints

def cloud_save(target):
    
    # camera object which defines ray source
    cam = bpy.data.objects['Camera']

    # save current view mode
    mode = bpy.context.area.type

    # set view mode to 3D to have all needed variables available
    bpy.context.area.type = "VIEW_3D"

    # get vectors which define view frustum of camera
    topRight,bottomRight,bottomLeft,topLeft = get_cam_frustum(cam)

    # number of pixels in X/Y direction
    resolutionX = int(bpy.context.scene.render.resolution_x * (bpy.context.scene.render.resolution_percentage / 100))
    resolutionY = int(bpy.context.scene.render.resolution_y * (bpy.context.scene.render.resolution_percentage / 100))

    # setup vectors to match pixels
    xRange = np.linspace(topLeft[0], topRight[0], resolutionX)
    yRange = np.linspace(topLeft[1], bottomLeft[1], resolutionY)
    xRange = xRange[230:250]
    yRange = yRange[170:190]
    
    # array to store hit information
    values = fill_array_with_none(xRange, yRange)

    # calculate origin
    matrixWorld = target.matrix_world
    matrixWorldInverted = matrixWorld.inverted()
    origin = matrixWorldInverted @ cam.matrix_world.translation

    indexX = 0
    indexY = 0

    # iterate over all X/Y coordinates
    for x in xRange:
        for y in yRange:
            # get current pixel vector from camera center to pixel
            pixelVector = Vector((x, y, topLeft[2]))

            # rotate that vector according to camera rotation
            pixelVector.rotate(cam.matrix_world.to_quaternion())

            # calculate direction vector
            destination = matrixWorldInverted @ (pixelVector + cam.matrix_world.translation) 
            direction = (destination - origin).normalized()

            # perform the actual ray casting
            hit, location, norm, face =  target.ray_cast(origin, direction)

            if hit:
                values[indexX,indexY] = (matrixWorld @ location)
            
            # update indices
            indexY += 1

        indexX += 1
        indexY = 0
    mesh, screenSpacePoints = create_mesh(values)

    # Create Object whose Object Data is our new mesh
    obj = bpy.data.objects.new('created object', mesh)

    #DEBUGGING    
    #bpy.context.active_object.rotation_euler[0] = math.radians(45)
    # Add *Object* to the scene, not the mesh
    scene = bpy.context.scene
    scene.collection.objects.link(obj)
    # reset view mode
    bpy.context.area.type = mode
    obj.data.update()

    # Select the new object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    return screenSpacePoints

def savePointCloudImage(screenSpacePoints, layer):
    obj = layer.objects.active
    verts = [vert.co for vert in obj.data.vertices]
    plain_verts = [vert.to_tuple() for vert in verts]
    array = np.array(plain_verts).reshape((20,20,3), order='F')
    array = array.reshape(1200)
    return array

def drawFaces(outputMatrix, outputVector, arrayToBeSend, layer):
    obj = layer.objects.active
    bm = bmesh.new()
    for i in range(400):
        bm.verts.new((arrayToBeSend[i*3], arrayToBeSend[i*3+1], arrayToBeSend[i*3+2]))
    bm.verts.ensure_lookup_table()

    index = 0
    for row in range(len(outputMatrix)):
        for col in range(row + 1,len(outputMatrix)):
            if outputVector[index] > 0:
                bm.edges.new([bm.verts[row], bm.verts[col]])
                #print("in if", index)
            index += 1
    bmesh.ops.edgenet_fill(bm, edges=bm.edges)
    if bpy.context.mode == 'EDIT_MESH':
        bmesh.update_edit_mesh(obj.data)
    else:
        bm.to_mesh(obj.data)

    obj.data.update()   

def main(args=None):
    context = zmq.Context()#  Socket to talk to server
    print("Connecting to hello world server…")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://localhost:5555")#  Do 10 requests, waiting each time for a response

    targetObj = bpy.context.view_layer.objects.active
    #targetObj = bpy.data.objects['object']
    #specify render resolution
    bpy.context.scene.render.resolution_x = 480
    bpy.context.scene.render.resolution_y = 360

    bpy.context.view_layer.update()
    print("Creating Cloud")
    #print(targetObj.data)
    screenSpacePoints = cloud_save(targetObj)
    arrayToBeSend = savePointCloudImage(screenSpacePoints, bpy.context.view_layer)
    print("Shape of input", arrayToBeSend.shape)
    socket.send(arrayToBeSend.tobytes())
    message = socket.recv() # 400*200 array

    outputVector = np.frombuffer(message, dtype=np.float32)
    print("Shape of output, " , outputVector.shape)
    outputMatrix = np.empty((400,400), dtype=np.float64)
    #print(outputVector)
    drawFaces(outputMatrix, outputVector, arrayToBeSend, bpy.context.view_layer)
    print("After draw facess")
    socket.close()

main()