""" blackfynnMesh.py
blackfynn mesh takes an input of ECG points and renders it to apply our data from Blackfynn to it

This file is modified from 'meshtype_2d_plate1.py' created by Richard Christie.

"""
import numpy as np
import math
from opencmiss.zinc.element import Element, Elementbasis
from opencmiss.zinc.field import Field
from opencmiss.zinc.node import Node
from opencmiss.zinc.glyph import Glyph
from mapclientplugins.greenstrainsstep.model.meshalignmentmodel import MeshAlignmentModel


class StrainMesh(MeshAlignmentModel):
    """
    StrainMesh is the central class used for generating the model and calculating strains on the model
    """

    def __init__(self, region, time_based_node_description):
        super(StrainMesh, self).__init__()
        self._mesh_group = []
        self._strain_graphics_point_attr = []
        self._field_element_group = None
        self._coordinates = None
        self._frame_index = 0
        self._strains_created = False
        ecg_region = region.findChildByName('strain_mesh')
        if ecg_region.isValid():
            region.removeChild(ecg_region)

        self._region = region.createChild('strain_mesh')
        self._time_based_node_description = time_based_node_description


    def generate_mesh(self):
        """
        generateMesh: This is where all points, elements, and colour fields relating to them are defined
        """
        coordinate_dimensions = 3
        # self.number_points = len(self._node_coordinate_list)

        # We currently find the number of elements by taking the square root of the number of given points
        elements_count_across = 7
        elements_count_up = 7
        use_cross_derivatives = 0

        # Set up our coordinate field
        field_module = self._region.getFieldmodule()
        field_module.beginChange()
        coordinates = field_module.createFieldFiniteElement(coordinate_dimensions)
        coordinates.setName('coordinates')
        coordinates.setManaged(True)
        coordinates.setTypeCoordinate(True)
        coordinates.setCoordinateSystemType(Field.COORDINATE_SYSTEM_TYPE_RECTANGULAR_CARTESIAN)
        coordinates.setComponentName(1, 'x')
        coordinates.setComponentName(2, 'y')
        if coordinate_dimensions == 3:
            coordinates.setComponentName(3, 'z')

        # Set up our node template
        nodes = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        node_template = nodes.createNodetemplate()
        node_template.defineField(coordinates)
        node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        if use_cross_derivatives:
            node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D2_DS1DS2, 1)

        mesh = field_module.findMeshByDimension(2)

        # Create our mesh subgroup
        fieldGroup = field_module.createFieldGroup()
        fieldElementGroup = fieldGroup.createFieldElementGroup(mesh)
        fieldElementGroup.setManaged(True)
        meshGroup = fieldElementGroup.getMeshGroup()

        # Define our interpolation
        bicubicHermiteBasis = field_module.createElementbasis(2, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        bilinearBasis = field_module.createElementbasis(2, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)

        # Set up our element templates
        eft = meshGroup.createElementfieldtemplate(bicubicHermiteBasis)
        eft_bi_linear = meshGroup.createElementfieldtemplate(bilinearBasis)
        if not use_cross_derivatives:
            for n in range(4):
                eft.setFunctionNumberOfTerms(n*4 + 4, 0)
        element_template = meshGroup.createElementtemplate()
        element_template.setElementShapeType(Element.SHAPE_TYPE_SQUARE)
        element_template.defineField(coordinates, -1, eft)

        # Create our spectrum colour field
        colour = field_module.createFieldFiniteElement(1)
        colour.setName('colour2')
        colour.setManaged(True)

        # add time support for colour field

        # Create node and element templates for our spectrum colour field
        node_template.defineField(colour)
        node_template.setValueNumberOfVersions(colour, -1, Node.VALUE_LABEL_VALUE, 1)
        element_template.defineField(colour, -1, eft_bi_linear)

        node_time_sequence = self._time_based_node_description['time_array']
        zinc_node_time_sequence = field_module.getMatchingTimesequence(node_time_sequence)
        node_template.setTimesequence(coordinates, zinc_node_time_sequence)

        first_node_number = 0
        # create nodes
        cache = field_module.createFieldcache()
        node_identifier = first_node_number
        zero = [0.0, 0.0, 0.0]
        i = 0
        for n2 in range(elements_count_up + 1):
            for n1 in range(elements_count_across + 1):

                node = nodes.createNode(node_identifier, node_template)
                cache.setNode(node)

                node_locations = self._time_based_node_description['{0}'.format(node_identifier)]
                # Assign the new node its position
                for index, time in enumerate(node_time_sequence):
                    cache.setTime(time)
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, node_locations[index])
                # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
                # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, dx_ds2)
                if use_cross_derivatives:
                    coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)

                node_identifier = node_identifier + 1
                i += 1

        # create elements
        element_node_list = []
        elementIdentifier = first_node_number
        mesh_group_list = []
        no2 = (elements_count_across + 1)
        for e2 in range(elements_count_up):
            for e1 in range(elements_count_across):
                element_node_list.append([])
                element = meshGroup.createElement(elementIdentifier, element_template)
                bni = e2 * no2 + e1 + first_node_number
                nodeIdentifiers = [bni, bni + 1, bni + no2, bni + no2 + 1]
                element_node_list[e1+e2*elements_count_across] = nodeIdentifiers
                result = element.setNodesByIdentifier(eft, nodeIdentifiers)
                result = element.setNodesByIdentifier(eft_bi_linear, nodeIdentifiers)
                elementIdentifier = elementIdentifier + 1
                element_group = field_module.createFieldElementGroup(mesh)
                temp_mesh_group = element_group.getMeshGroup()
                temp_mesh_group.addElement(element)
                mesh_group_list.append(element_group)

        # Set fields for later access
        self._mesh_group = meshGroup
        self._mesh_group_list = mesh_group_list
        self._element_node_list = element_node_list
        self._field_element_group = fieldElementGroup
        self._coordinates = coordinates

        field_module.endChange()

    def display_strain_at(self, time):

        for i, mg in enumerate(self._mesh_group_list):
            # strain = self.calculate_strain_in_element_xi(element_node_list[i], 0)
            # # scaled_eigvectors = self.get_sized_eigvectors(strain)
            # e_vals, e_vecs = np.linalg.eig(strain)

            e_vals, e_vecs = self.calculate_strain_in_element_xi(self._element_node_list[i], self._frame_index, time)
            if np.iscomplex(e_vecs).any():
                x = 0
            if self._strains_created is False:
                self.create_display_strains(e_vecs, e_vals, mg)
            else:
                self.update_display_strains(e_vecs, e_vals, i)
        self._strains_created = True
        print('Finished updating strains')


    def get_sized_eigvectors(self, E):

        e_vals, e_vecs = np.linalg.eig(E)
        sizedvectors = []
        for i, _ in enumerate(e_vals):
            sizedvectors.append(list(e_vals[i] * e_vecs[:, i]))
        return np.vstack(sizedvectors).tolist(), e_vals

    def create_display_strains(self, strain_vectors, strain_values, mesh_group):
        scene = self._region.getScene()
        fm = self._region.getFieldmodule()
        fm.beginChange()
        materialModule = scene.getMaterialmodule()
        coordinates = self._coordinates
        coordinates = coordinates.castFiniteElement()
        pointattrT = []
        for i in range(2):

            strain_graphics = scene.createGraphicsPoints()
            strain_graphics.setFieldDomainType(Field.DOMAIN_TYPE_MESH_HIGHEST_DIMENSION)
            strain_graphics.setCoordinateField(coordinates)
            strain_graphics.setSubgroupField(mesh_group)
            pointattr = strain_graphics.getGraphicspointattributes()
            pointattr.setGlyphShapeType(Glyph.SHAPE_TYPE_ARROW_SOLID)
            pointattr.setGlyphRepeatMode(Glyph.REPEAT_MODE_MIRROR)

            pointattr.setBaseSize([0.00,0.005,0.005])
            ss = fm.createFieldConstant(strain_vectors[i].tolist())
            pointattr.setOrientationScaleField(ss)
            pointattr.setSignedScaleField(fm.createFieldConstant(strain_values[i]))
            strain_graphics.setMaterial(materialModule.findMaterialByName('red'))
            pointattr.setScaleFactors([.1, 0.0, 0.0])
            pointattrT.append(pointattr)
        self._strain_graphics_point_attr.append(pointattrT)
        # strain_graphics.setName('displayStrains')

        # Create a point attribute arrray so that we can modify the size later easily

        fm.endChange()

    def update_display_strains(self, strain_vectors, strain_values, index):
        scene = self._region.getScene()
        fm = self._region.getFieldmodule()
        fm.beginChange()
        for i in range(2):
            ss = fm.createFieldConstant(strain_vectors[i].tolist())
            point_attr = self._strain_graphics_point_attr[index][i]
            point_attr.setOrientationScaleField(ss)
            point_attr.setSignedScaleField(fm.createFieldConstant(strain_values[i]))

    def convert_dict_to_array(self, dictionary):
        array = []
        for key in dictionary:
            if key is not 'time_array':
                array.append(dictionary[key])
        return array

    def calculate_strain(self, points, points_dash):
        F = np.linalg.solve(points, points_dash)
        C = F.T @ F
        E = .5 * (C - np.identity(3))
        return E

    def calculate_strain_in_element_xi(self, element, ref_t, t):

        nodes = self._time_based_node_description

        # xl and yl are our un adjusted element directions. We will use these create a local 3D coordinates for the element

        # xl = point 2 of our element - point 1
        xl = np.array( nodes[str(element[1])][ref_t] ) - np.array( nodes[str(element[0])][ref_t] )
        yl = np.array( nodes[str(element[0])][ref_t] ) - np.array( nodes[str(element[2])][ref_t] )

        xi = xl / np.linalg.norm(xl)
        norm = np.cross(yl, xl)
        zi = norm / np.linalg.norm(norm)
        yi = np.cross(zi, xi)
        yi = yi / np.linalg.norm(yi)
        TT = np.vstack([xi, yi])
        # Transormation Matrix TM will be used to convert between coordinate systems
        # https://stackoverflow.com/questions/19621069/3d-rotation-matrix-rotate-to-another-reference-system
        TM = TT

        points = [nodes[str(element[1])][ref_t], nodes[str(element[2])][ref_t], nodes[str(element[3])][ref_t]]
        points_dash = [nodes[str(element[1])][t], nodes[str(element[2])][t], nodes[str(element[3])][t]]

        # Calculate strain using E = .5(Ftrnsp . F - I ) https://www.continuummechanics.org/greenstrain.html
        F = np.linalg.solve(points, points_dash)
        C = F.T @ F
        E = .5 * (C - np.identity(3))

        # Transform stain matrix to our local coordinates
        Exi = TM @ E @ TM.T

        # Calculate strain eigenvalues and eigenvectors in the local xi coordinates
        e_vals = [0,0,0]
        e_vals[0:2], e_vecsxi = np.linalg.eig(Exi)
        e_vecs = np.eye(3, 3)
        e_vecs[:, 0] = TM.T @ e_vecsxi[:,0]
        e_vecs[:, 1] = TM.T @ e_vecsxi[:,1]
        e_vecs[:, 2] = [0,0,0]

        return np.array(e_vals), e_vecs.T


    def set_strain_reference_frame(self, frame_index):
        self._frame_index = frame_index

    def drawMesh(self):

        scene = self._region.getScene()
        fm = self._region.getFieldmodule()

        coordinates = self._coordinates
        coordinates = coordinates.castFiniteElement()

        materialModule = scene.getMaterialmodule()

        axes = scene.createGraphicsPoints()
        pointattr = axes.getGraphicspointattributes()
        pointattr.setGlyphShapeType(Glyph.SHAPE_TYPE_AXES_XYZ)
        pointattr.setBaseSize([1.0, 1.0, 1.0])
        axes.setMaterial(materialModule.findMaterialByName('grey50'))
        axes.setName('displayAxes')
        axes.setVisibilityFlag(True)

        lines = scene.createGraphicsLines()
        lines.setCoordinateField(coordinates)
        lines.setName('displayLines2')
        lines.setMaterial(materialModule.findMaterialByName('blue'))

        nodePoints = scene.createGraphicsPoints()
        nodePoints.setFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodePoints.setCoordinateField(coordinates)
        nodePoints.setMaterial(materialModule.findMaterialByName('blue'))
        nodePoints.setVisibilityFlag(True)

        nodePointAttr = nodePoints.getGraphicspointattributes()
        nodePointAttr.setGlyphShapeType(Glyph.SHAPE_TYPE_SPHERE)
        nodePointAttr.setBaseSize([.005, .005, .005])
        cmiss_number = fm.findFieldByName('cmiss_number')
        nodePointAttr.setLabelField(cmiss_number)

        surfaces = scene.createGraphicsSurfaces()
        surfaces.setCoordinateField(coordinates)
        surfaces.setMaterial(materialModule.findMaterialByName('trans_blue'))
        surfaces.setVisibilityFlag(True)

        # colour = fm.findFieldByName('colour2')
        # colour = colour.castFiniteElement()

        # Set attributes for our mesh
        scene.endChange()
