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
from opencmiss.zinc.spectrum import Spectrumcomponent
from opencmiss.zinc.graphics import Graphics
from mapclientplugins.greenstrainsstep.model.meshalignmentmodel import MeshAlignmentModel
from opencmiss.zinc.status import OK as ZINC_OK
from opencmiss.zinc.streamregion import StreaminformationRegion


class StrainMesh(MeshAlignmentModel):
    """
    BlackfynnMesh is the central point for generating the model for our mesh and drawing it
    """

    def __init__(self, region, time_based_node_description):
        super(StrainMesh, self).__init__()
        self._mesh_group = []
        self._field_element_group = None
        self._coordinates = None
        self._frame_index = 0

        ecg_region = region.findChildByName('strain_mesh')
        if ecg_region.isValid():
            region.removeChild(ecg_region)

        self._region = region.createChild('strain_mesh')
        self._scene = self._region.getScene()
        self._time_based_node_description = time_based_node_description


    def generate_mesh(self):
        """
        generateMesh: This is where all points, elements, and colour fields relating to them are defined
        """
        # self.loadModel(self._region)
        coordinate_dimensions = 3
        # self.number_points = len(self._node_coordinate_list)

        # We currently find the number of elements by taking the square root of the number of given points
        elements_count_across = 7
        elements_count_up = 7
        use_cross_derivatives = 0

        # ctx = self._region.getContext()
        # logger = ctx.getLogger()
        elementsCount1 = 1
        elementsCount2 = 1
        elementsCount3 = 1

        # Set up our coordinate field
        field_module = self._region.getFieldmodule()
        fm = field_module
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

        nodes = fm.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        nodetemplate = nodes.createNodetemplate()
        nodetemplate.defineField(coordinates)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)

        nodetemplate.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)


        mesh = fm.findMeshByDimension(3)
        from scaffoldmaker.utils.eftfactory_tricubichermite import eftfactory_tricubichermite
        tricubichermite = eftfactory_tricubichermite(mesh, 0)
        eft = tricubichermite.createEftBasic()

        elementtemplate = mesh.createElementtemplate()
        elementtemplate.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        result = elementtemplate.defineField(coordinates, -1, eft)

        cache = fm.createFieldcache()

        # create nodes
        nodeIdentifier = 1
        x = [0.0, 0.0, 0.0]
        x1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        x2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0], [1.0, 0.5, 1.0],
              [0.0, 2.5, 1.5], [1.0, 2.5, 1.5]]
        points = []
        dx_ds1 = [1.0 / elementsCount1, 0.0, 0.0]
        dx_ds2 = [0.0, 1.0 / elementsCount2, 0.0]
        dx_ds3 = [0.0, 0.0, 1.0 / elementsCount3]
        zero = [0.0, 0.0, 0.0]
        node_time_sequence = [0,2]
        zinc_node_time_sequence = field_module.getMatchingTimesequence(node_time_sequence)
        nodetemplate.setTimesequence(coordinates, zinc_node_time_sequence)

        first_node_number = 0

        # create nodes
        nodeIdentifier = 1
        x = [0.0, 0.0, 0.0]
        x1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
              [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        F = np.array([[1,.5,-1],
             [.5,1,0],
             [0,0,2]])

        x2 = (np.array(x1) @ F).tolist()
        # x2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0], [1.0, 0.5, 1.0],
        #       [0.0, 2.5, 1.5], [1.0, 2.5, 1.5]]
        points = []
        dx_ds1 = [1.0 / elementsCount1, 0.0, 0.0]
        dx_ds2 = [0.0, 1.0 / elementsCount2, 0.0]
        dx_ds3 = [0.0, 0.0, 1.0 / elementsCount3]
        zero = [0.0, 0.0, 0.0]

        cache = field_module.createFieldcache()
        nodeIdentifier = 1
        for i, xp in enumerate(x1):
            node = nodes.createNode(nodeIdentifier, nodetemplate)
            cache.setTime(0)
            cache.setNode(node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xp)
            # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
            # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, dx_ds2)
            # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, dx_ds3)
            # if useCrossDerivatives:
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, zero)
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS2DS3, 1, zero)
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D3_DS1DS2DS3, 1, zero)

            cache.setTime(2)
            cache.setNode(node)
            coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x2[i])
            # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
            # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, dx_ds2)
            # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, dx_ds3)
            # if useCrossDerivatives:
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, zero)
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS2DS3, 1, zero)
            #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D3_DS1DS2DS3, 1, zero)
            nodeIdentifier = nodeIdentifier + 1
            self.node = node
        # create elements
        elementIdentifier = 1
        no2 = (elementsCount1 + 1)
        no3 = (elementsCount2 + 1) * no2
        for e3 in range(elementsCount3):
            for e2 in range(elementsCount2):
                for e1 in range(elementsCount1):
                    element = mesh.createElement(elementIdentifier, elementtemplate)
                    bni = e3 * no3 + e2 * no2 + e1 + 1
                    nodeIdentifiers = [bni, bni + 1, bni + no2, bni + no2 + 1, bni + no3, bni + no3 + 1,
                                       bni + no2 + no3, bni + no2 + no3 + 1]
                    result = element.setNodesByIdentifier(eft, nodeIdentifiers)
                    elementIdentifier = elementIdentifier + 1

        fm.endChange()
        # Set up our node template
        # nodes = field_module.findNodesetByFieldDomainType(Field.DOMAIN_TYPE_NODES)
        # node_template = nodes.createNodetemplate()
        # node_template.defineField(coordinates)
        # node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_VALUE, 1)
        # node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS1, 1)
        # node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS2, 1)
        # node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D_DS3, 1)
        # if use_cross_derivatives:
        #     node_template.setValueNumberOfVersions(coordinates, -1, Node.VALUE_LABEL_D2_DS1DS2, 1)
        #
        # mesh = field_module.findMeshByDimension(3)
        # # mesh = mesh.getMasterMesh()
        # # Create our mesh subgroup
        # # fieldGroup = field_module.createFieldGroup()
        # # fieldElementGroup = fieldGroup.createFieldElementGroup(mesh)
        # # fieldElementGroup.setManaged(True)
        # # meshGroup = fieldElementGroup.getMeshGroup()
        #
        # # Define our interpolation
        # bicubicHermiteBasis = field_module.createElementbasis(3, Elementbasis.FUNCTION_TYPE_CUBIC_HERMITE)
        # # bilinearBasis = field_module.createElementbasis(2, Elementbasis.FUNCTION_TYPE_LINEAR_LAGRANGE)
        #
        # # Set up our element templates
        # eft = mesh.createElementfieldtemplate(bicubicHermiteBasis)
        # # eft_bi_linear = mesh.createElementfieldtemplate(bilinearBasis)
        # # if not use_cross_derivatives:
        # #     for n in range(4):
        # #         eft.setFunctionNumberOfTerms(n*4 + 4, 0)
        # element_template = mesh.createElementtemplate()
        # element_template.setElementShapeType(Element.SHAPE_TYPE_CUBE)
        # element_template.defineField(coordinates, -1, eft)
        #
        # # Create our spectrum colour field
        # # colour = field_module.createFieldFiniteElement(1)
        # #         # colour.setName('colour2')
        # #         # colour.setManaged(True)
        #
        # # add time support for colour field
        #
        # # Create node and element templates for our spectrum colour field
        # # node_template.defineField(colour)
        # # node_template.setValueNumberOfVersions(colour, -1, Node.VALUE_LABEL_VALUE, 1)
        # # element_template.defineField(colour, -1, eft_bi_linear)
        #
        # node_time_sequence = [0,2]
        # zinc_node_time_sequence = field_module.getMatchingTimesequence(node_time_sequence)
        # node_template.setTimesequence(coordinates, zinc_node_time_sequence)
        #
        # first_node_number = 0
        # elementsCount1 = elementsCount2 = elementsCount3 = 1
        #
        # # create nodes
        # nodeIdentifier = 1
        # x = [0.0, 0.0, 0.0]
        # x1 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0],
        #       [0.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        # x2 = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.5], [1.0, 2.0, 0.5], [0.0, 0.5, 1.0], [1.0, 0.5, 1.0],
        #       [0.0, 2.5, 1.5], [1.0, 2.5, 1.5]]
        # points = []
        # dx_ds1 = [1.0 / elementsCount1, 0.0, 0.0]
        # dx_ds2 = [0.0, 1.0 / elementsCount2, 0.0]
        # dx_ds3 = [0.0, 0.0, 1.0 / elementsCount3]
        # zero = [0.0, 0.0, 0.0]
        #
        # cache = field_module.createFieldcache()
        # nodeIdentifier = 1
        # for i, xp in enumerate(x1):
        #     node = nodes.createNode(nodeIdentifier, node_template)
        #     cache.setTime(0)
        #     cache.setNode(node)
        #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, xp)
        #     # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
        #     # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, dx_ds2)
        #     # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, dx_ds3)
        #     # if useCrossDerivatives:
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, zero)
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS2DS3, 1, zero)
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D3_DS1DS2DS3, 1, zero)
        #
        #     cache.setTime(2)
        #     cache.setNode(node)
        #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_VALUE, 1, x2[i])
        #     # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS1, 1, dx_ds1)
        #     # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS2, 1, dx_ds2)
        #     # coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D_DS3, 1, dx_ds3)
        #     # if useCrossDerivatives:
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS2, 1, zero)
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS1DS3, 1, zero)
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D2_DS2DS3, 1, zero)
        #     #     coordinates.setNodeParameters(cache, -1, Node.VALUE_LABEL_D3_DS1DS2DS3, 1, zero)
        #     nodeIdentifier = nodeIdentifier + 1
        #     self.node = node
        #
        #
        # # create elements
        # elementIdentifier = 1
        # no2 = (elementsCount1 + 1)
        # no3 = (elementsCount2 + 1) * no2
        # for e3 in range(elementsCount3):
        #     for e2 in range(elementsCount2):
        #         for e1 in range(elementsCount1):
        #             element = mesh.createElement(elementIdentifier, element_template)
        #             bni = e3 * no3 + e2 * no2 + e1 + 1
        #             nodeIdentifiers = [bni, bni + 1, bni + no2, bni + no2 + 1, bni + no3, bni + no3 + 1,
        #                                bni + no2 + no3, bni + no2 + no3 + 1]
        #             result = element.setNodesByIdentifier(eft, nodeIdentifiers)
        #             elementIdentifier = elementIdentifier + 1



        # Set fields for later access
        # self._mesh_group = meshGroup
        # self._element_list = element_list
        # self._mesh_group_list = mesh_group_list
        # self._element_node_list = element_node_list
        # self._field_element_group = fieldElementGroup
        self._coordinates = coordinates
        field_module.endChange()
        self._strain_graphics_point_attr = []
        self.strain_in_zinc(1)

    def convert_dict_to_array(self, dictionary):
        array = []
        for key in dictionary:
            if key is not 'time_array':
                array.append(dictionary[key])
        return array

    def strain_in_zinc(self, time):
        fieldmodule = self._region.getFieldmodule()
        scene = self._scene
        coordinates = self._coordinates
        coordinates = coordinates.castFiniteElement()
        scene.beginChange()
        E = fieldmodule.findFieldByName("E") # Will not be valid on first run
        principal_strain_direction = []
        principal_strain = []
        if E.isValid():
            for i in range(3):
                principal_strain.append(fieldmodule.findFieldByName("principal_strain{:}".format(i + 1)))
                principal_strain_direction.append(
                    fieldmodule.findFieldByName("principal_strain{:}_direction".format(i + 1)))
        else:
            fieldmodule.beginChange()
            reference_coordinates = fieldmodule.createFieldTimeLookup(coordinates, fieldmodule.createFieldConstant(0))
            F = fieldmodule.createFieldGradient(coordinates, reference_coordinates)
            F_transpose = fieldmodule.createFieldTranspose(3, F)
            identity3 = fieldmodule.createFieldConstant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            C = fieldmodule.createFieldMatrixMultiply(3, F_transpose, F)
            E2 = C - identity3
            E = E2 * fieldmodule.createFieldConstant(0.5)
            E.setName("E")
            E.setManaged(True)

            # -- transform E into fibre coordinates --
            fibre_axes = fieldmodule.createFieldFibreAxes(fieldmodule.createFieldConstant(0), reference_coordinates) # why do we pass in a 0?
            E1 = fieldmodule.createFieldMatrixMultiply(3, fibre_axes, E)
            fat = fieldmodule.createFieldTranspose(3, fibre_axes)
            E_fibre = fieldmodule.createFieldMatrixMultiply(3, E1, fat)

            # -- find principle strains in fibre coordinates --
            principal_strains = fieldmodule.createFieldEigenvalues(E_fibre)
            principal_strains.setName("principal_strains")
            principal_strains.setManaged(True)
            principal_strain_vectors = fieldmodule.createFieldEigenvectors(principal_strains)

            # -- converting back to global coordinates? --
            deformed_principal_strain_vectors = fieldmodule.createFieldMatrixMultiply(3, principal_strain_vectors,
                                                                                      F_transpose)
            # -- splitting the matrix into components --
            deformed_principal_strain_vector = [ \
                fieldmodule.createFieldMatrixMultiply(1, fieldmodule.createFieldConstant([1.0, 0.0, 0.0]),
                                                      deformed_principal_strain_vectors), \
                fieldmodule.createFieldMatrixMultiply(1, fieldmodule.createFieldConstant([0.0, 1.0, 0.0]),
                                                      deformed_principal_strain_vectors), \
                fieldmodule.createFieldMatrixMultiply(1, fieldmodule.createFieldConstant([0.0, 0.0, 1.0]),
                                                      deformed_principal_strain_vectors)]

            # -- get principle strain directions --
            for i in range(3):
                direction = fieldmodule.createFieldNormalise(deformed_principal_strain_vector[i])
                direction.setName("principal_strain{:}_direction".format(i + 1))
                direction.setManaged(True)
                principal_strain_direction.append(direction)
                strain = fieldmodule.createFieldComponent(principal_strains, i + 1)
                strain.setName("principal_strain{:}".format(i + 1))
                strain.setManaged(True)
                principal_strain.append(strain)
            # Calculate the deformed fibre axes
            # fibre_axes = fieldmodule.createFieldFibreAxes(fibres, rc_reference_coordinates)
            # fibre_axes.setName("fibre_axes")
            # fibre_axes.setManaged(True)
            # deformed_fibre_axes = fieldmodule.createFieldMatrixMultiply(3, fibre_axes, F_transpose)
            # deformed_fibre_axes.setName("deformed_fibre_axes")
            # deformed_fibre_axes.setManaged(True)
            fieldmodule.endChange()
        spectrummodule = scene.getSpectrummodule()
        strainSpectrum = spectrummodule.findSpectrumByName("strain")
        if not strainSpectrum.isValid():
            spectrummodule.beginChange()
            strainSpectrum = spectrummodule.createSpectrum()
            strainSpectrum.setName("strain")
            strainSpectrum.setManaged(True)
            # red when negative
            spectrumComponent1 = strainSpectrum.createSpectrumcomponent()
            spectrumComponent1.setColourMappingType(Spectrumcomponent.COLOUR_MAPPING_TYPE_RED)
            spectrumComponent1.setRangeMinimum(-1.0)
            spectrumComponent1.setRangeMaximum(0.0)
            spectrumComponent1.setExtendAbove(False)
            spectrumComponent1.setColourMinimum(1.0)
            spectrumComponent1.setColourMaximum(1.0)
            # blue when positive
            spectrumComponent2 = strainSpectrum.createSpectrumcomponent()
            spectrumComponent2.setColourMappingType(Spectrumcomponent.COLOUR_MAPPING_TYPE_BLUE)
            spectrumComponent2.setRangeMinimum(0.0)
            spectrumComponent2.setRangeMaximum(1.0)
            spectrumComponent2.setExtendBelow(False)
            spectrumComponent2.setColourMinimum(1.0)
            spectrumComponent2.setColourMaximum(1.0)
            # this adds some green to the blue above so not too dark
            spectrumComponent3 = strainSpectrum.createSpectrumcomponent()
            spectrumComponent3.setColourMappingType(Spectrumcomponent.COLOUR_MAPPING_TYPE_GREEN)
            spectrumComponent3.setRangeMinimum(0.0)
            spectrumComponent3.setRangeMaximum(1.0)
            spectrumComponent3.setExtendBelow(False)
            spectrumComponent3.setColourMinimum(0.5)
            spectrumComponent3.setColourMaximum(0.5)
            spectrummodule.endChange()
        # visualise the strain vectors with mirrored glyphs
        for i in range(3):
            points = scene.createGraphicsPoints()
            points.setFieldDomainType(Field.DOMAIN_TYPE_MESH3D)
            points.setCoordinateField(coordinates)
            points.setDataField(principal_strain[i])
            points.setSpectrum(strainSpectrum)
            attr = points.getGraphicspointattributes()
            attr.setGlyphShapeType(Glyph.SHAPE_TYPE_CONE)
            attr.setGlyphRepeatMode(Glyph.REPEAT_MODE_MIRROR)
            attr.setBaseSize([1.0, 1.0, 1.0])
            attr.setOrientationScaleField(principal_strain_direction[i])
            attr.setSignedScaleField(principal_strain[i])
            attr.setScaleFactors([2.0, 0.0, 0.0])
        scene.endChange()


    def set_strain_reference_frame(self, frame_index):
        self._frame_index = frame_index


    def old_strain(self):
        xi = xl / np.linalg.norm(xl)
        norm = np.cross(yl, xl)
        zi = norm / np.linalg.norm(norm)
        yi = np.cross(zi, xi)
        yi = yi / np.linalg.norm(yi)
        TT = np.vstack([xi, yi])
        # Transormation Matrix TM will be used to convert between coordinate systems
        # https://stackoverflow.com/questions/19621069/3d-rotation-matrix-rotate-to-another-reference-system
        TM = TT

        Exi = TM @ E @ TM.T
        e_vals = [0, 0, 0]
        e_vals[0:2], e_vecsxi = np.linalg.eig(Exi)
        e_vecs = np.eye(3, 3)
        e_vecs[:, 0] = TM.T @ e_vecsxi[:, 0]
        e_vecs[:, 1] = TM.T @ e_vecsxi[:, 1]
        e_vecs[:, 2] = [0, 0, 0]


    def drawMesh(self):
        scene = self._scene
        fm = self._region.getFieldmodule()
        scene.beginChange()

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
        # surfaces.setRenderPolygonMode(Graphics.RENDER_POLYGON_MODE_WIREFRAME)
        surfaces.setMaterial(materialModule.findMaterialByName('trans_blue'))
        surfaces.setExterior(True)
        surfaces.setVisibilityFlag(True)
        # colour = fm.findFieldByName('colour2')
        # colour = colour.castFiniteElement()

        # Set attributes for our mesh
        scene.endChange()

    def loadModel(self, region):
        '''
        Read time-varying deforming heart model.
        Define strains fields and make some graphics to visualise them.
        '''
        sir = region.createStreaminformationRegion()
        path = "C:\\Users\jkho021\Downloads\Sound Cloud\examples\\a\deforming_heart\cmiss_input\\"
        sir.createStreamresourceFile(path + "reference_heart.exnode")
        sir.createStreamresourceFile(path + "reference_heart.exelem")
        for i in range(51):
            filename = 'heart{:0>4}.exnode'.format(i)
            fr = sir.createStreamresourceFile(path+filename)
            sir.setResourceAttributeReal(fr, StreaminformationRegion.ATTRIBUTE_TIME, i / 50.0)
        sir.createStreamresourceFile(path+"heart.exelem")
        result = region.read(sir)
        if result != ZINC_OK:
            print("failed to read")
            return False
        scene = region.getScene()
        timekeepermodule = scene.getTimekeepermodule()
        timekeeper = timekeepermodule.getDefaultTimekeeper()
        timekeeper.setMinimumTime(0.0)
        timekeeper.setMaximumTime(1.0)
        timekeeper.setTime(0.0)

        scene.beginChange()
        scene.removeAllGraphics()
        fieldmodule = region.getFieldmodule()

        # Where are Coordinaets and reference coordinates coming from?
        coordinates = fieldmodule.findFieldByName("coordinates")
        reference_coordinates = fieldmodule.findFieldByName("reference_coordinates")

        # Where is fibre coming from?
        fibres = fieldmodule.findFieldByName("fibres")

        lines = scene.createGraphicsLines()
        lines.setCoordinateField(coordinates)
        surfaces = scene.createGraphicsSurfaces()
        surfaces.setName("surfaces")
        surfaces.setCoordinateField(coordinates)
        surfaces.setExterior(True)
        surfaces.setElementFaceType(Element.FACE_TYPE_XI3_0)

        principal_strain_direction = []
        principal_strain = []
        E = fieldmodule.findFieldByName("E")
        if E.isValid():
            for i in range(3):
                principal_strain.append(fieldmodule.findFieldByName("principal_strain{:}".format(i + 1)))
                principal_strain_direction.append(
                    fieldmodule.findFieldByName("principal_strain{:}_direction".format(i + 1)))
        else:
            fieldmodule.beginChange()
            rc_reference_coordinates = fieldmodule.createFieldCoordinateTransformation(reference_coordinates)
            rc_coordinates = fieldmodule.createFieldCoordinateTransformation(coordinates)
            F = fieldmodule.createFieldGradient(rc_coordinates, rc_reference_coordinates)
            F_transpose = fieldmodule.createFieldTranspose(3, F)
            identity3 = fieldmodule.createFieldConstant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
            C = fieldmodule.createFieldMatrixMultiply(3, F_transpose, F)
            E2 = C - identity3
            E = E2 * fieldmodule.createFieldConstant(0.5)
            E.setName("E")
            E.setManaged(True)
            principal_strains = fieldmodule.createFieldEigenvalues(E)
            principal_strains.setName("principal_strains")
            principal_strains.setManaged(True)
            principal_strain_vectors = fieldmodule.createFieldEigenvectors(principal_strains)
            deformed_principal_strain_vectors = fieldmodule.createFieldMatrixMultiply(3, principal_strain_vectors,
                                                                                      F_transpose)
            # should be easier than this to get several components:
            deformed_principal_strain_vector = [ \
                fieldmodule.createFieldMatrixMultiply(1, fieldmodule.createFieldConstant([1.0, 0.0, 0.0]),
                                                      deformed_principal_strain_vectors), \
                fieldmodule.createFieldMatrixMultiply(1, fieldmodule.createFieldConstant([0.0, 1.0, 0.0]),
                                                      deformed_principal_strain_vectors), \
                fieldmodule.createFieldMatrixMultiply(1, fieldmodule.createFieldConstant([0.0, 0.0, 1.0]),
                                                      deformed_principal_strain_vectors)]
            for i in range(3):
                direction = fieldmodule.createFieldNormalise(deformed_principal_strain_vector[i])
                direction.setName("principal_strain{:}_direction".format(i + 1))
                direction.setManaged(True)
                principal_strain_direction.append(direction)
                strain = fieldmodule.createFieldComponent(principal_strains, i + 1)
                strain.setName("principal_strain{:}".format(i + 1))
                strain.setManaged(True)
                principal_strain.append(strain)
            # Calculate the deformed fibre axes
            fibre_axes = fieldmodule.createFieldFibreAxes(fibres, rc_reference_coordinates)
            fibre_axes.setName("fibre_axes")
            fibre_axes.setManaged(True)
            deformed_fibre_axes = fieldmodule.createFieldMatrixMultiply(3, fibre_axes, F_transpose)
            deformed_fibre_axes.setName("deformed_fibre_axes")
            deformed_fibre_axes.setManaged(True)
            fieldmodule.endChange()

        spectrummodule = scene.getSpectrummodule()
        strainSpectrum = spectrummodule.findSpectrumByName("strain")
        if not strainSpectrum.isValid():
            spectrummodule.beginChange()
            strainSpectrum = spectrummodule.createSpectrum()
            strainSpectrum.setName("strain")
            strainSpectrum.setManaged(True)
            # red when negative
            spectrumComponent1 = strainSpectrum.createSpectrumcomponent()
            spectrumComponent1.setColourMappingType(Spectrumcomponent.COLOUR_MAPPING_TYPE_RED)
            spectrumComponent1.setRangeMinimum(-1.0)
            spectrumComponent1.setRangeMaximum(0.0)
            spectrumComponent1.setExtendAbove(False)
            spectrumComponent1.setColourMinimum(1.0)
            spectrumComponent1.setColourMaximum(1.0)
            # blue when positive
            spectrumComponent2 = strainSpectrum.createSpectrumcomponent()
            spectrumComponent2.setColourMappingType(Spectrumcomponent.COLOUR_MAPPING_TYPE_BLUE)
            spectrumComponent2.setRangeMinimum(0.0)
            spectrumComponent2.setRangeMaximum(1.0)
            spectrumComponent2.setExtendBelow(False)
            spectrumComponent2.setColourMinimum(1.0)
            spectrumComponent2.setColourMaximum(1.0)
            # this adds some green to the blue above so not too dark
            spectrumComponent3 = strainSpectrum.createSpectrumcomponent()
            spectrumComponent3.setColourMappingType(Spectrumcomponent.COLOUR_MAPPING_TYPE_GREEN)
            spectrumComponent3.setRangeMinimum(0.0)
            spectrumComponent3.setRangeMaximum(1.0)
            spectrumComponent3.setExtendBelow(False)
            spectrumComponent3.setColourMinimum(0.5)
            spectrumComponent3.setColourMaximum(0.5)
            spectrummodule.endChange()

        # visualise the strain vectors with mirrored glyphs
        for i in range(3):
            points = scene.createGraphicsPoints()
            points.setFieldDomainType(Field.DOMAIN_TYPE_MESH3D)
            points.setCoordinateField(coordinates)
            points.setDataField(principal_strain[i])
            points.setSpectrum(strainSpectrum)
            attr = points.getGraphicspointattributes()
            attr.setGlyphShapeType(Glyph.SHAPE_TYPE_CONE)
            attr.setGlyphRepeatMode(Glyph.REPEAT_MODE_MIRROR)
            attr.setBaseSize([0.0, 1.0, 1.0])
            attr.setOrientationScaleField(principal_strain_direction[i])
            attr.setSignedScaleField(principal_strain[i])
            attr.setScaleFactors([20.0, 0.0, 0.0])

        scene.endChange()
        return True
