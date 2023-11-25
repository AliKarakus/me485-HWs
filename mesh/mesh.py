from __future__ import print_function
import scipy as sp
import numpy as np
import sys
import os

from .parse import parse

class mesh:
    """
mesh
    Parse mesh file and create all connectivities and geometric factors 
    """
#-------------------------------------------------------------------------------------------------#
    def setup(self, mshfilename):
        self.Nelements  = 0;  self.NBelements = 0; 
        self.NFaces     = 0;  self.NBFaces    = 0;
        self.Nnodes     = 0;  self.NBnodes    = 0;
        self.Element    = {}; self.Node       = {}; self.Face       = {}
        self.BToV       = {}; self.BType      = {}

        self.weightMethod = 'distance'
        # self.weightMethod = 'volume'


        self.BCMap = {
        1: {'name': 'WALL',    'gtype': 'DRICHLET', 'dtype': 'DRICHLET'},
        2: {'name': 'INFLOW',  'gtype': 'DRICHLET', 'dtype': 'DRICHLET'},
        3: {'name': 'OUTFLOW', 'gtype': 'NEUMANN',  'dtype': 'NEUMANN' },
        4: {'name': 'SYMMETRY','gtype': 'NEUMANN',  'dtype': 'NEUMANN' },
        }

        self.elmInfo = {
            1: {'name': 'line', 'nfaces': 2, 'nverts': 2, 'ftype':0,
            'fnodes': [0, 1]},
            2: {'name': 'tri', 'nfaces': 3, 'nverts': 3, 'ftype': 1,
            'fnodes': [[0, 1], [1, 2], [2, 0]]},
            3: {'name': 'quad', 'nfaces': 4, 'nverts': 4, 'ftype': 1,
            'fnodes': [[0, 1], [1, 2], [2, 3], [3, 0]]},
            4: {'name': 'penta', 'nfaces': 5, 'nverts': 5, 'ftype':1,
            'fnodes': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 0]]},
            5: {'name': 'tet', 'nfaces': 4, 'nverts': 4, 'ftype': 2,
            'fnodes': [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]},
            6: {'name': 'hex', 'nfaces': 6, 'nverts': 8, 'ftype': 3,
            'fnodes': [[0, 1, 2, 3], [0, 1, 4, 5], [1, 2, 5, 6], 
            [2, 3, 6, 7], [0, 3, 4, 7], [4, 5, 6, 7]]},
            }
        # start with creating a parse to read mesh
        self.parser = parse(mshfilename)
        self.Nverts = self.parser.Nnodes 

        # Set Maximum Dimesion in the Mesh i.e. problem dimension
        self.dim = self.get_max_dim()

        self.elementRule()
        self.nodeRule()
        self.parser.parse()

        # Obtain Element, Node and Node Data 
        self.connectNodes()
        # print('nodes are connected')
        self.connectElements()
        # print('element are connected')
        self.connectFaces()
        # print('face are connected')

        self.connectNodeBC()
        self.getNodeWights()
#-------------------------------------------------------------------------------------------------#
    
#-------------------------------------------------------------------------------------------------#
    def connectFaces(self):
        # make sure Element is constructed
        if(len(self.Element)==0):
            self.connectElements()
        # make sure Node is constructed
        if(len(self.Node)==0):
            self.connectNodes()
        
        # Create unified face indexing
        usedfaces = {}
        for elm, einfo in self.Element.items():
            etype   = einfo['elementType'] 
            nfaces  = self.elmInfo[etype]['nfaces']
            usedfaces[elm] = []
            for f in range(nfaces):
                usedfaces[elm].append(0)

        sk= 0; 
        for elm, info in self.Element.items():
            etype   = info['elementType'] 
            nfaces  = self.elmInfo[etype]['nfaces']
            for f in range(nfaces):
                eM = elm; fM = f
                if(usedfaces[eM][fM] == 0):
                    self.Face[sk] = {}
                    eP = info['neighElement'][fM]
                    fP = info['neighFace'][fM]
                    bc = info['boundary'][fM]

                    self.Face[sk]['owner']     = eM
                    self.Face[sk]['neigh']     = eP
                    self.Face[sk]['ownerFace'] = fM
                    self.Face[sk]['neighFace'] = info['neighFace'][fM]

                    self.Face[sk]['nodes']     = info['nodes'][self.elmInfo[etype]['fnodes'][fM]]

                    self.Face[sk]['normal']    = info['normal'][fM]
                    self.Face[sk]['center']    = info['fcenter'][fM]

                    self.Face[sk]['boundary']  = info['boundary'][fM]
                    self.Face[sk]['bcid']      = info['bcid'][fM]

                    self.Face[sk]['weight']    = float(info['weight'][fM])
                    self.Face[sk]['area']      = float(info['area'][fM])
                    usedfaces[eP][fP] = 1
                    sk = sk+1

        self.NBFaces = 0 
        for elm, info in self.Element.items():
            bcdata = info['boundary']
            for bc in bcdata:
                if(bc!=0):
                    self.NBFaces = self.NBFaces + 1

        self.NFaces = len(self.Face)

        sk= 0; 
        for elm, info in self.Element.items():
            etype   = info['elementType'] 
            nfaces  = self.elmInfo[etype]['nfaces']
            for f in range(nfaces):
                if(usedfaces[elm][f] == 0):
                    info['facemap'].append(sk)
                    sk = sk+1
                else:
                    info['facemap'].append(sk)
#-------------------------------------------------------------------------------------------------#
# This routine creates element -> vertex connectivity 
# mesh.EToE[element_id] gives list of vertices connected to this element 
#-------------------------------------------------------------------------------------------------#
    def elementRule(self):

        def cndEToV(etag, etype, pgrp, nodes):
            return True

        # Condition for EtoV
        def actEToV(etag, etype, pgrp, nodes):
            if(self.parser.physical_group_dims[pgrp] == self.dim):
                self.Element[self.Nelements] = {}
                self.Element[self.Nelements]['nodes'] = nodes -1 # make zero indexed            
                self.Element[self.Nelements]['elementType'] = etype                
                self.Element[self.Nelements]['elementTag']  = etag                
                self.Nelements = self.Nelements + 1
           
            # This reads co-dimension 1 objects
            if(self.parser.physical_group_dims[pgrp] == (self.dim-1)):
                self.BToV[self.NBelements]  = nodes -1 
                self.BType[self.NBelements] =  pgrp # make zero indexed
                self.NBelements = self.NBelements + 1

        self.Element= self.sort_dict(self.Element)
        self.BToV   = self.sort_dict(self.BToV)
        self.BType  = self.sort_dict(self.BType)

        self.NBFaces = len(self.BToV)

        self.parser.add_elements_rule(cndEToV, actEToV)

#-------------------------------------------------------------------------------------------------#
# This routine creates element -> vertex connectivity 
# mesh.EToE[element_id] gives list of vertices connected to this element 
#-------------------------------------------------------------------------------------------------#
    def nodeRule(self):
        # if(len(self.VX)==0):
        if(len(self.Node)==0):
            # first create a condition and action for parsing
            def condition(tag,x,y,z,physgroups):
                return True

            # Condition for EtoV
            def action(tag, x,y,z):
                self.Node[tag-1] = {}
                self.Node[tag-1]['coord']       = sp.array([x,y,z])
                self.Node[tag-1]['element']     = []
                self.Node[tag-1]['weight']      = []
                self.Node[tag-1]['face_weight'] = []
                self.Node[tag-1]['boundary']    = []
                self.Node[tag-1]['fboundary']   = []
                self.Node[tag-1]['face']        = []
            
            self.parser.add_nodes_rule(condition, action)
            
            self.Node = self.sort_dict(self.Node)
            self.Nnodes = len(self.Node) 

#-------------------------------------------------------------------------------------------------#
# This routine creates vertex -> element connectivity 
# mesh.EToV[vertex_id] gives list of elements connected to this vertex 
#-------------------------------------------------------------------------------------------------#
    def connectNodes(self):
        for elm, info in self.Element.items():
            vrtx = info['nodes']
            for vid in vrtx:
                if elm not in self.Node[vid]['element']:
                    self.Node[vid]['element'].append(elm)

        self.Node = self.sort_dict(self.Node)

#-------------------------------------------------------------------------------------------------#
    def connectNodeBC(self):

        for elm, info in self.Element.items():
            nodes = info['nodes']
            etype = info['elementType']
            nfaces = self.elmInfo[etype]['nfaces']
            for f in range(nfaces):
                bc = info['boundary'][f]
                if bc!=0 :
                    nodeids = self.elmInfo[etype]['fnodes'][f][:]
                    fnodes = nodes[nodeids[:]]
                    for i in range(len(fnodes)):
                        self.Node[fnodes[i]]['boundary'] = bc 


        for face, info in self.Face.items():
            nodes = info['nodes']
            for vrt in range(len(nodes)):
                self.Node[nodes[vrt]]['face'].append(face)
                self.Node[nodes[vrt]]['fboundary'].append(info['boundary'])

#-------------------------------------------------------------------------------------------------#
# This routine creates vertex -> element connectivity 
# mesh.EToV[vertex_id] gives list of elements connected to this vertex 
#-------------------------------------------------------------------------------------------------#
    # def getNodeWights(self):
    #     for vrtx, info in self.Node.items():
    #         elements = info['element']
    #         bc       = info['boundary']
    #         xv       = info['coord']

    #         if(bc==0):
    #             print(info)
    #             for elm in range(len(elements)):
    #                 eid = elements[elm]
    #                 xe  = self.Element[eid]['ecenter']
    #                 wi  = 1.0 / sp.linalg.norm(xv-xe)**2
    #                 self.Node[vrtx]['weight'].append(wi)



    #         # # first  
    #         # for elm in range(len(elements)):
    #         #     eid = elements[elm]
    #         #     xe  = self.Element[eid]['ecenter']
    #         #     wi  = 1.0 / sp.linalg.norm(xv-xe)**2
    #         #     self.Node[vrtx]['weight'].append(wi)

    #         # if(bc != 0):
    #         #     bcs = info['fboundary']
    #         #     for f in range(len(bcs)):
    #         #         if(bcs[f] != 0):
    #         #             faceid = info['face'][f]
    #         #             xf     = self.Face[faceid]['center']
    #         #             wi     = 1.0 / sp.linalg.norm(xv-xf)**3
    #         #             self.Node[vrtx]['weight'].append(wi)

    #         total_weight              = sp.sum(self.Node[vrtx]['weight'])
    #         self.Node[vrtx]['weight'] =  self.Node[vrtx]['weight']/total_weight
#-------------------------------------------------------------------------------------------------#
# This routine creates vertex -> element connectivity 
# mesh.EToV[vertex_id] gives list of elements connected to this vertex 
#-------------------------------------------------------------------------------------------------#
    def getNodeWights(self):
        sk = 0
        for vrtx, info in self.Node.items():
            elements = info['element']
            bc       = info['boundary']
            xv       = info['coord']

            # print(vrtx, bc)

            if(bc):
                wi = 1.0; 
                self.Node[vrtx]['weight'].append(wi)
                self.Node[vrtx]['bcid'] = sk
                sk += 1
            else:
                for elm in range(len(elements)):
                    eid = elements[elm]
                    xe  = self.Element[eid]['ecenter']
                    wi  = 1.0 / sp.linalg.norm(xv-xe)**2
                    self.Node[vrtx]['weight'].append(wi)

                total_weight              = sp.sum(self.Node[vrtx]['weight'])
                self.Node[vrtx]['weight'] =  self.Node[vrtx]['weight']/total_weight

        self.NBVertices = sk
#-------------------------------------------------------------------------------------------------#
    def connectElements(self):
 # Check if EToV and EToE are ready
        if(len(self.Element)==0):
            self.connect_EToV()

        Nfaces = 0
        # Compute Total faces in the grid
        for elm in self.Element.keys():
            etype  = self.Element[elm]['elementType'] 
            Nfaces = Nfaces + self.elmInfo[etype]['nfaces']

        self.NTotalFaces = Nfaces

        rows = sp.full(3*Nfaces, -1)
        cols = sp.full(3*Nfaces, -1)
        vals = sp.full(3*Nfaces, -1)

        sk = 0; ek = 0; 
        for elm in self.Element.keys():
            etype      = self.Element[elm]['elementType']  
            nfaces     = self.elmInfo[etype]['nfaces']
            face_nodes = self.elmInfo[etype]['fnodes']
            for face in range(nfaces):
                vn     = self.Element[elm]['nodes'] 
                verts  = vn[face_nodes[face][:]]           
                for vrt in range(len(verts)):
                    rows[sk] = ek
                    cols[sk] = verts[vrt]
                    vals[sk] = 1 
                    sk = sk + 1                
                ek = ek+1

        rows = rows[:sk]
        cols = cols[:sk]
        vals = vals[:sk]


        NfacesSofar = 0
        CON     = np.zeros((self.NTotalFaces,2), int)
        FSTART  = np.zeros((self.Nelements,1), int)
        sk = 0
        for elm in self.Element.keys():
            etype      = self.Element[elm]['elementType']  
            nfaces     = self.elmInfo[etype]['nfaces']
            FSTART[elm] = sk
            for face in range(nfaces):
                CON[sk, 0]  = elm
                CON[sk, 1]  = sk
                sk = sk + 1


        BFACE = np.zeros((self.NBelements*5, 2), dtype=int) - 1

        sk = 0
        for bface in range(self.NBelements):
            verts = self.BToV[bface]
            for v in range(len(verts)):
                BFACE[sk, 0] = bface
                BFACE[sk, 1] = verts[v]
                sk = sk+1
        # BFACE = np.resize(BFACE, [sk, 2])
        BFACE = BFACE[0:sk,:]
        
        SpFToV = sp.sparse.coo_matrix( (vals, (rows, cols)), shape = (Nfaces, self.Nverts), dtype=int)

        # This holds self connections as well
        SpFToF = SpFToV @ SpFToV.transpose()  #- 2*sp.sparse.identity(Nfaces)

        # Face - To - Face connection with global Ids
        faceId1 = sp.sparse.find((SpFToF == 2))[:][0]
        faceId2 = sp.sparse.find((SpFToF == 2))[:][1]
        sk = 0; bcid = 0
        for elm in self.Element.keys():
            etype      = self.Element[elm]['elementType']   
            nfaces     = self.elmInfo[etype]['nfaces']
            self.Element[elm]['neighElement'] = []
            self.Element[elm]['neighFace']    = []
            self.Element[elm]['boundary']     = []
            self.Element[elm]['bcid']         = []
            self.Element[elm]['facemap']      = []
            for face in range(nfaces):
                faceM1 = faceId1[sk]
                faceM2 = faceId1[sk+1]
                # this means that there is no bc face here
                if( faceM1==faceM2):
                    faceP1 = faceId2[sk]
                    faceP2 = faceId2[sk+1]
                    if(faceM1 == faceP1):
                        inds = np.where(CON[:,1] == faceP2)
                        self.Element[elm]['neighElement'].append(CON[inds, 0].item())
                        self.Element[elm]['neighFace'].append(faceP2 - FSTART[CON[inds, 0]].item())
                        self.Element[elm]['boundary'].append(0)
                        self.Element[elm]['bcid'].append(-1)
                    else:
                        inds = np.where(CON[:,1] == faceP1)
                        self.Element[elm]['neighElement'].append(CON[inds, 0].item())
                        self.Element[elm]['neighFace'].append(faceP1 - FSTART[CON[inds, 0]].item())
                        self.Element[elm]['boundary'].append(0)
                        self.Element[elm]['bcid'].append(-1)
                    sk = sk + 2
                else: # boundary face
                    self.Element[elm]['neighElement'].append(elm)
                    self.Element[elm]['neighFace'].append(face)
                    face_nodes = self.elmInfo[etype]['fnodes']
                    fverts      = self.Element[elm]['nodes'][face_nodes[face][:]]

                    inds1 = np.where(BFACE[:,1] == fverts[0])
                    inds2 = np.where(BFACE[:,1] == fverts[1])
                    elmbc = np.intersect1d(BFACE[inds1,0], BFACE[inds2,0]).item()

                    # elmbc = self.find_in_dict(self.BToV, self.Element[elm]['nodes'][face_nodes[face][:]])
                    self.Element[elm]['boundary'].append( self.BType[elmbc])
                    self.Element[elm]['bcid'].append(bcid)
                    bcid = bcid +1
                    sk = sk + 1
        
         # Compute Element Centers
        for elm in self.Element.keys():
            etype     = self.Element[elm]['elementType']
            verts  = self.Element[elm]['nodes']
            xc, vol   = self.compute_volume(verts, etype)

            self.Element[elm]['ecenter'] = xc
            self.Element[elm]['volume'] = vol
            
        # Compute Face Centers
        for elm in self.Element.keys():
            etype  = self.Element[elm]['elementType']
            nfaces = self.elmInfo[etype]['nfaces'] 

            self.Element[elm]['fcenter']  = []
            self.Element[elm]['normal']  = []
            self.Element[elm]['area'] = []

            verts = self.Element[elm]['nodes']
            for face in range(nfaces): 
                face_nodes = self.elmInfo[etype]['fnodes']
                fverts = self.Element[elm]['nodes'][face_nodes[face][:]]
                ftype = self.elmInfo[etype]['ftype']

                fxc, sA  = self.compute_volume(fverts, ftype)
                normal  = self.compute_normal(fverts, etype)

                self.Element[elm]['area'].append(sA)
                self.Element[elm]['normal'].append(normal)
                self.Element[elm]['fcenter'].append(fxc)
        #compute wight for the face
        for elm, info in self.Element.items():
            etype  = info['elementType']
            nfaces = self.elmInfo[etype]['nfaces'] 
            self.Element[elm]['weight'] = sp.zeros((nfaces,1), float)
            for f in range(nfaces):
                if(info['boundary'][f] == 0 ):                    
                    #normal distance based wights
                    if(self.weightMethod == 'distance'):
                        #neigbor element of the face
                        eP     = self.Element[elm]['neighElement'][f]
                        dfdxF  = self.Element[eP]['ecenter'] -  info['fcenter'][f]
                        dfdxE  = self.Element[eP]['ecenter'] -  info['ecenter']

                        normf = sp.linalg.norm(dfdxF)**1
                        norme = sp.linalg.norm(dfdxE)**1

                        weight = normf/norme
                    if(self.weightMethod == 'volume'):
                        vM     = info['volume']
                        eP     = self.Element[elm]['neighElement'][f]
                        vP     = self.Element[eP]['volume']
                        weight = vM / (vP + vM)

                    # self.Element[elm]['weight'][f] = weight
                    self.Element[elm]['weight'][f] = weight
                else:
                    self.Element[elm]['weight'][f] = 0.5

#-------------------------------------------------------------------------------------------------#
    def find_in_dict(self, dict, myvals):
            for key, vals in dict.items():
                ind = sp.flatnonzero(sp.in1d(sp.array(vals), sp.array(myvals)))
                if(len(ind)==len(myvals)):
                    return key
#-------------------------------------------------------------------------------------------------#
    def find_in_list(self, list1, list2):
        sk = 0
        for row in list1:
            if(sorted(row)==sorted(list2)):
                return sk
            sk = sk+1
#-------------------------------------------------------------------------------------------------#
    def report(self):
        nline = 0;  ntri  = 0; nquad = 0
        npenta = 0; ntet  = 0; nhex  = 0
        for elm in self.Element.keys():
            etype  = self.Element[elm]['elementType']
            if(self.elmInfo[etype]['name'] == 'line'):
                nline = nline +1
            if(self.elmInfo[etype]['name'] == 'tri'):
                ntri = ntri +1
            if(self.elmInfo[etype]['name'] == 'quad'):
                nquad = nquad +1
            if(self.elmInfo[etype]['name'] == 'penta'):
                npenta = npenta +1
            if(self.elmInfo[etype]['name'] == 'tet'):
                ntet = ntet +1
            if(self.elmInfo[etype]['name'] == 'hex'):
                nhex = nhex +1

        print("--------------------------- Reporting Elements ----------------------------"
            .center(os.get_terminal_size().columns))

        print('{0:<40} : {1:<4d}'.format("Number of Line Elements", nline))
        print('{0:<40} : {1:<4d}'.format("Number of Triangular Elements", ntri))
        print('{0:<40} : {1:<4d}'.format("Number of Quadrilateral Elements", nquad))
        print('{0:<40} : {1:<4d}'.format("Number of Pentagonal Elements", npenta))
        print('{0:<40} : {1:<4d}'.format("Number of Tetrehedral Elements", ntet))
        print('{0:<40} : {1:<4d}'.format("Number of Hexehedral Elements", nhex))

        print("--------------------------- Reporting Mesh Info ----------------------------"
            .center(os.get_terminal_size().columns))

        # self.NBFaces = 0 
        # for elm, info in self.Element.items():
        #     bcdata = info['boundary']
        #     for bc in bcdata:
        #         if(bc>0):
        #             self.NBFaces = self.NBFaces + 1

        # print('{0:<40} : {1:<4d}'.format("Number of Boundary Faces", self.NBFaces))

#-------------------------------------------------------------------------------------------------#
    def get_max_dim(self):
        dim = 0
        for g in self.parser.physical_groups:
            group_dim = self.parser.physical_group_dims[g]
            dim = max(dim, group_dim)

        return dim
#-------------------------------------------------------------------------------------------------#   
    def sort_dict(self, unsorted_dict):
        # sorted_dict = unsorted_dict
        Keys = list(unsorted_dict.keys())
        Keys.sort()
        sorted_dict = {i: unsorted_dict[i] for i in Keys}
        
        return sorted_dict


#-------------------------------------------------------------------------------------------------#   
    def print_dict(self, dict):
        for n in dict.keys():
            print('%d %s' %(n, dict[n])) 

#-------------------------------------------------------------------------------------------------#   
    def compute_normal(self, vrtx, etype):
        #line
        if(self.elmInfo[etype]['ftype'] == 1):
            v1 = self.Node[vrtx[0]]['coord'] 
            v2 = self.Node[vrtx[1]]['coord']
            nn = sp.cross( v2 - v1, sp.array([0, 0, 1]))
            return nn/sp.linalg.norm(nn)
        
        # tri
        if(self.elmInfo[etype]['ftype'] == 2):
            v1 = self.Node[vrtx[0]]['coord'] 
            v2 = self.Node[vrtx[1]]['coord']
            v3 = self.Node[vrtx[2]]['coord']
            nn = 0.5*sp.cross( v2 - v1, v3-v1)
            return  nn/sp.linalg.norm(nn)
        
        # quad
        if(self.elmInfo[etype]['ftype'] == 3):
            v1 = self.Node[vrtx[0]]['coord'] 
            v2 = self.Node[vrtx[1]]['coord']
            v3 = self.Node[vrtx[2]]['coord']
            v4 = self.Node[vrtx[3]]['coord']
            nn = 0.5*sp.cross( v2 - v1, v3-v1) + 0.5*sp.cross(v2-v3,v4-v3)
            return  nn/sp.linalg.norm(nn)
#-------------------------------------------------------------------------------------------------#   
    def compute_volume(self, vrtx, etype):
        if(self.elmInfo[etype]['name'] == 'line'):
            Tri = sp.zeros((2,3), float)
            Tri[0][:] = self.Node[vrtx[0]]['coord'] 
            Tri[1][:] = self.Node[vrtx[1]]['coord'] 

            area  = sp.linalg.norm(sp.cross( Tri[0][:] - Tri[1][:], sp.array([0, 0, 1])))
            xc    = sp.mean(Tri, axis=0)  
            return xc, area

        if(self.elmInfo[etype]['name'] == 'tri'):
            Tri = sp.zeros((3,3), float)
            Tri[0][:] = self.Node[vrtx[0]]['coord'] 
            Tri[1][:] = self.Node[vrtx[1]]['coord'] 
            Tri[2][:] = self.Node[vrtx[2]]['coord']

            area = self.computeSubArea(Tri, etype)
            xc   = self.computeSubCenter(Tri, etype) 
            return xc, area
        if(self.elmInfo[etype]['name'] == 'quad'):
            Tri = sp.zeros((3,3), float)
            xv1 = self.Node[vrtx[0]]['coord'] 
            xv2 = self.Node[vrtx[1]]['coord'] 
            xv3 = self.Node[vrtx[2]]['coord'] 
            xv4 = self.Node[vrtx[3]]['coord'] 
            xcg = (xv1 + xv2 + xv3 + xv4)/4.0

            Tri[0][:] = xv1[:]; Tri[1][:] = xv2[:]; Tri[2][:] = xcg[:]
            at1 = self.computeSubArea(Tri, etype)
            xc1 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv2[:]; Tri[1][:] = xv3[:]; Tri[2][:] = xcg[:]
            at2 = self.computeSubArea(Tri, etype)
            xc2 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv3[:]; Tri[1][:] = xv4[:]; Tri[2][:] = xcg[:]
            at3 = self.computeSubArea(Tri, etype)
            xc3 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv4[:]; Tri[1][:] = xv1[:]; Tri[2][:] = xcg[:]
            at4 = self.computeSubArea(Tri, etype)
            xc4 = self.computeSubCenter(Tri, etype) 

            ar = at1 + at2 + at3 + at4
            xc = (xc1*at1 + xc2*at2 +xc3*at3 +xc4*at4)/ar 
            return xc, ar
        if(self.elmInfo[etype]['name'] == 'penta'):
            Tri = sp.zeros((3,3))
            xv1 = self.Node[vrtx[0]]['coord'] 
            xv2 = self.Node[vrtx[1]]['coord'] 
            xv3 = self.Node[vrtx[2]]['coord'] 
            xv4 = self.Node[vrtx[3]]['coord'] 
            xv5 = self.Node[vrtx[4]]['coord']
            xcg = (xv1+xv2+xv3+xv4+xv5)/5.0

            Tri[0][:] = xv1[:]; Tri[1][:] = xv2[:]; Tri[2][:] = xcg[:]
            at1 = self.computeSubArea(Tri, etype)
            xc1 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv2[:]; Tri[1][:] = xv3[:]; Tri[2][:] = xcg[:]
            at2 = self.computeSubArea(Tri, etype)
            xc2 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv3[:]; Tri[1][:] = xv4[:]; Tri[2][:] = xcg[:]
            at3 = self.computeSubArea(Tri, etype)
            xc3 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv4[:]; Tri[1][:] = xv5[:]; Tri[2][:] = xcg[:]
            at4 = self.computeSubArea(Tri, etype)
            xc4 = self.computeSubCenter(Tri, etype) 

            Tri[0][:] = xv5[:]; Tri[1][:] = xv1[:]; Tri[2][:] = xcg[:]
            at5 = self.computeSubArea(Tri, etype)
            xc5 = self.computeSubCenter(Tri, etype) 

            ar = at1 + at2 + at3 + at4 + at5
            xc = (xc1*at1 + xc2*at2 + xc3*at3 + xc4*at4 + xc5*at5)/ar 
            return xc, ar
#-------------------------------------------------------------------------------------------------#   
    def computeSubArea(self, Tri, etype):
        if(len(Tri) == 3):
            v1 = Tri[0][:]; v2 = Tri[1][:]; v3 = Tri[2][:]
            return sp.linalg.norm( 0.5*sp.cross(v2 - v1, v3-v1))

    def computeSubCenter(self, Tri, etype):
          return sp.mean(Tri, axis=0)