import bpy
from mathutils import Vector, Quaternion
import numpy as np
import bmesh
import time
import math
import mathutils
import csv
import collections
from scipy import spatial
from PIL import Image

def render_save(path):
    bpy.context.scene.render.filepath = path
    bpy.ops.render.render(write_still=True)

def move_cam_randomly(start, end, focus_point=mathutils.Vector((0.0, 0.0, 0.0))):
    camera = bpy.data.objects['Camera']
    sign = (-1)**np.random.randint(2)
    tX = np.random.uniform(start,end)*sign

    sign = (-1)**np.random.randint(2)
    tY = np.random.uniform(start, end)*sign

    sign = (-1)**np.random.randint(2)
    tZ = np.random.uniform(start, end)*sign
    camera.location = (tX, tY, tZ)
    
    bpy.data.objects['Light'].location = camera.location
    
    looking_direction = camera.location - focus_point
    rot_quat = looking_direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def outputGenerator(path,layer,screenSpacePoints):
    obj = layer.objects.active
    bm = bmesh.new()
    bm.from_object(obj, bpy.context.evaluated_depsgraph_get())
    verts = [vert.co for vert in obj.data.vertices]
    bm.verts.ensure_lookup_table()
    # coordinates as tuples
    plain_verts = [vert.to_tuple() for vert in verts]
    
    tri = spatial.Delaunay(screenSpacePoints, qhull_options="Qt")
    adjacency_matrix = [ [0]*400 for i in range(400)]
    adjacency_vector = [0]*(400*200)


    print("tricimplices", len(tri.simplices))
    for tuple in tri.simplices:
        p0 = tuple[0]//20 + (tuple[0] % 20) * 20
        p1 = tuple[1]//20 + (tuple[1] % 20) * 20
        p2 = tuple[2]//20 + (tuple[2] % 20) * 20
        adjacency_matrix[p0][p1] = 1 # edge tuple[0]-tuple[1]
        adjacency_matrix[p1][p0] = 1
        adjacency_matrix[p0][p2] = 1 # edge tuple[0]-tuple[2]
        adjacency_matrix[p2][p0] = 1
        adjacency_matrix[p1][p2] = 1 # edge tuple[1]-tuple[2]
        adjacency_matrix[p2][p1] = 1

    index = 0

    for row in range(len(adjacency_matrix)):
        for col in range(row + 1,len(adjacency_matrix)):
                adjacency_vector[index] = adjacency_matrix[row][col]
                index += 1
    np.savetxt(path+".vector",adjacency_vector)

    # for vert in bm.verts:
    #     if len(vert.link_edges) == 0:
    #         bm.verts.remove(vert)

    # if bpy.context.mode == 'EDIT_MESH':
    #     bmesh.update_edit_mesh(obj.data)
    # else:
    #     bm.to_mesh(obj.data)

    # obj.data.update()


    
    
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
        #print(index, 'index')
        # add new vertex
        screenSpacePoints.append(index)
        bm.verts.new((location[0], location[1], location[2]))
        bm.verts.ensure_lookup_table()
    #    print(len(screenSpacePoints), "screeeeen")
    # make the bmesh the object's mesh
    bm.to_mesh(mesh)  
    bm.free()  # always do this when finished

    # We're done setting up the mesh values, update mesh object and 
    # let Blender do some checks on it
    mesh.update()
    mesh.validate()

    return mesh, screenSpacePoints
#=============================================================
def cloud_save(path, target):
    
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
    
    print("xrange is " , xRange)
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
    #print(values)
    mesh, screenSpacePoints = create_mesh(values)

    # Create Object whose Object Data is our new mesh
    obj = bpy.data.objects.new('created object', mesh)
    #faceobj = bpy.data.objects.new('created face object', facemesh)
    #faceobj.data.update()

    #DEBUGGING    
    #bpy.context.active_object.rotation_euler[0] = math.radians(45)
    # Add *Object* to the scene, not the mesh
    scene = bpy.context.scene
    scene.collection.objects.link(obj)
    #scene.collection.objects.link(faceobj)
    # reset view mode
    bpy.context.area.type = mode
    obj.data.update()

    # Select the new object and make it active
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    

    '''
    with open(path+'.csv','w', newline='') as out:
        header = ['X', 'Y', 'Z']
        csv_out=csv.writer(out)
        # writing data row-wise into the csv file
        csv_out.writerow(header)
        for row in plain_verts:
            csv_out.writerow(row)
    '''
    return screenSpacePoints

def savePointCloudImage(path,screenSpacePoints, layer):
    obj = layer.objects.active
    verts = [vert.co for vert in obj.data.vertices]
    plain_verts = [vert.to_tuple() for vert in verts]
    #print(*plain_verts, sep='\n')
    array = np.array(plain_verts).reshape((20,20,3), order='F')
    #print(*array, sep='\n')
    array = array.reshape(1200)
    np.savetxt(path+".nk", array)
    #ar = np.loadtxt(path+".nk")
    #print("=================================")
    #ar = ar.reshape((20,20,3))
    #print(*ar, sep='\n')



def main(args):
    
    renderCount = args[0]
    path = args[1]
    start = args[2]
    end = args[3]
    targetObj = bpy.context.view_layer.objects.active
    targetObj = bpy.data.objects['Cube']
    #specify render resolution
    bpy.context.scene.render.resolution_x = 480
    bpy.context.scene.render.resolution_y = 360
    for i in range(0,renderCount):
        move_cam_randomly(start, end, targetObj.location)
        #render_save(path+"inputs\\"+str(i))
        bpy.context.view_layer.update()
        print("Creating Cloud")
        #print(targetObj.data)
        screenSpacePoints = cloud_save(path+str(i), targetObj)
        print("Saving Point Cloud Image")
        savePointCloudImage(path+"inputs/"+str(i), screenSpacePoints, bpy.context.view_layer)
        print("hiding")    
        targetObj.hide_set(True)
        outputGenerator(path+"outputs/"+str(i), bpy.context.view_layer, screenSpacePoints)
        #render_save(path+"outputs/"+str(i))
        targetObj.hide_set(False)
        bpy.ops.object.delete()
        bpy.ops.object.select_all(action='DESELECT')
        targetObj.select_set(True)
        bpy.context.view_layer.objects.active = targetObj
main( [1000, "C:/Dataset/", 2,6])

print("Done.")