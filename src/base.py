# import scipy as sp
import numpy as np
from mesh import mesh
import pyfiglet as pfg
import os

class base(mesh):
    """
base manager
    this class manages field and setups
    """
#-------------------------------------------------------------------------------------------------#
    def __init__(self, meshfile):
       
        print("----------------------------------------------------------------------"
            .center(int(os.get_terminal_size().columns * .3)))
        text1 = pfg.figlet_format('-> M E   4 8 5 <-', font = 'slant') 
        # introtext = pfg.figlet_format('--  ME 485 --', font = '3-d') 
        # introtext = pfg.figlet_format('--  ME 485 --', font = 'alphabet') 
        # introtext = pfg.figlet_format('--  ME 485 --', font = 'doh') 
        # introtext = pfg.figlet_format('--  ME 485 --', font = 'letters') 
        # introtext = pfg.figlet_format('--  ME 485 --', font = 'bubble') 
        # introtext = pfg.figlet_format('--  ME 485 --', font = 'digital') 
        print(text1)


        print("----------------------------------------------------------------------"
            .center(int(os.get_terminal_size().columns * .3)))

       
        self.setup(meshfile)
        print('gmsh reader: mesh is connected')

        print("----------------------------------------------------------------------"
            .center(int(os.get_terminal_size().columns * .3)))


#-------------------------------------------------------------------------------------------------#
    def createEfield(self, Nfields):

        return np.zeros((self.Nelements, Nfields), float)

#-------------------------------------------------------------------------------------------------#
    def createVfield(self, Nfields, dim):

        return np.zeros((self.Nverts, Nfields, dim), float)

#-------------------------------------------------------------------------------------------------#
    def createFfield(self, Nfields, dim):
        return np.zeros((self.NFaces, Nfields, dim), float)
#-------------------------------------------------------------------------------------------------#                
    def cell2Node(self, Qe, Qb, method):
        shape = Qe.shape; dim = 1
        if(len(shape) == 3):
            dim = shape[2]

        Nfields = shape[1]

        Qv = np.zeros((self.Nverts, Nfields, dim), float)
        if(Qe.shape[1] != Qv.shape[1]):
            print('Cell2Node: dimesion of the matrtices are not equal: exiting')
            return -1
        else:

            if(method == 'average'):
                for vrt, info in self.Node.items():
                    elements = info['element']
                    bc       = info['boundary']
                    xv       = info['coord']
                    Qv[vrt]  = 0.0
                    sk = 0
                    if(bc):
                        # qb     = Qb[self.Face[global_face_Id]['bcid']]
                        qb = Qb[info['bcid']]
                        Qv[vrt] = qb
                    else:
                        for e in range(len(elements)):
                            eid = elements[e]
                            wi = self.Node[vrt]['weight'][sk]
                            Qv[vrt] = Qv[vrt] + wi*Qe[eid]
                            sk = sk+1
        return Qv                  
#-------------------------------------------------------------------------------------------------#
    def plotVTU(self, fileName, Q):
        fp = open(fileName, 'w')

        fp.write("<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"BigEndian\">\n")
        fp.write("  <UnstructuredGrid>\n")
        fp.write("    <Piece NumberOfPoints=\"%d\" NumberOfCells=\"%d\">\n" % (self.Nverts, self.Nelements))

        # write out nodes
        fp.write("      <Points>\n")
        fp.write("        <DataArray type=\"Float32\" NumberOfComponents=\"3\" Format=\"ascii\">\n")

        # compute plot node coordinates on the fly
        for vrtx, info in self.Node.items():
            coord = info['coord']
            fp.write("       ")
            fp.write("%g %g %g\n" % (coord[0], coord[1], coord[2]))

        fp.write("        </DataArray>\n")
        fp.write("      </Points>\n")

#-------------------------------------------------------------------------------------------------#   
        fp.write("      <PointData Scalars=\"scalars\">\n");
        
        if (len(Q) != 0):
            if(Q.shape[2] == 1):
                #write out pressure
                fp.write("        <DataArray type=\"Float32\" Name=\"field\" Format=\"ascii\">\n")
                for vrt, info in self.Node.items():
                    fp.write("       ")
                    fp.write("%g\n" %Q[vrt, 0, 0])
                fp.write("        </DataArray>\n")

            if(Q.shape[2] == 2):
                #write out pressure
                fp.write("        <DataArray type=\"Float32\" Name=\"field\" NumberOfComponents=\"2\" Format=\"ascii\">\n")               
                for vrt, info in self.Node.items():            
                    fp.write("       ")
                    fp.write("%g %g \n" %(Q[vrt][0][0], Q[vrt][0][1]))
                fp.write("        </DataArray>\n")

            if(Q.shape[2] == 3):
                print('right here')
                #write out pressure
                fp.write("        <DataArray type=\"Float32\" Name=\"field\" NumberOfComponents=\"3\" Format=\"ascii\">\n")
                for vrt, info in self.Node.items():            
                    fp.write("       ")
                    fp.write("%g %g %g \n" %(Q[vrt][:][0], Q[vrt][:][1], Q[vrt][:][2]))
                fp.write("        </DataArray>\n")
     


        fp.write("      </PointData>\n")
#-------------------------------------------------------------------------------------------------#
        fp.write("    <Cells>\n")
        fp.write("      <DataArray type=\"Int32\" Name=\"connectivity\" Format=\"ascii\">\n")
        for elm, info in self.Element.items():
            vrtx = info['nodes']
            fp.write("       ")
            for v in vrtx:
                fp.write("%d " % (v))
            fp.write("\n")

        fp.write("        </DataArray>\n")

        fp.write("        <DataArray type=\"Int32\" Name=\"offsets\" Format=\"ascii\">\n")
        cnt = 0
        for elm,info in self.Element.items():
            etype = info['elementType']
            nvrts = len(info['nodes'])
            cnt += nvrts
            fp.write("       ")
            fp.write("%d\n" % cnt)

        fp.write("       </DataArray>\n")

        fp.write("       <DataArray type=\"Int32\" Name=\"types\" Format=\"ascii\">\n")
        for elm, info in self.Element.items():
            etype = info['elementType']
            if(self.elmInfo[etype]['name'] == 'tri'):
                fp.write("5\n")
            if(self.elmInfo[etype]['name'] == 'quad'):
                fp.write("9\n")
            # if(self.elmInfo[etype]['name'] == 'tet'):
            #     fp.write("10\n")
            # if(self.elmInfo[etype]['name'] == 'hex'):
            #     fp.write("12\n")

        fp.write("        </DataArray>\n")
        fp.write("      </Cells>\n")
        fp.write("    </Piece>\n")
        fp.write("  </UnstructuredGrid>\n")
        fp.write("</VTKFile>\n")
        fp.close()









