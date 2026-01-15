-- this add-on function is a minimalistic scene content exporter, meant as an example.

if (sim.msgbox_return_yes==sim.msgBox(sim.msgbox_type_info,sim.msgbox_buttons_yesno,"Minimalistic Exporter","This add-on function is a minimalistic scene content exporter, meant as an example. The scene content will be exported to the 'exportedContent' folder, erasing its previous content. If a single object or model is selected, then only the selection will be exported. Do you want to proceed?")) then
	local directoryName="exportedContent"
	local fileName="sceneObjects.txt"
	local meshFormat=4 -- 0=OBJ, 1=DXF, 4=binary STL
	local exportIndividualShapeComponents=true

	local appPath=sim.getStringParameter(sim.stringparam_application_path)
	local exportDir=appPath.."/"..directoryName
	local extension
	if meshFormat==0 then
		extension="obj"
	end
	if meshFormat==1 then
		extension="dxf"
	end
	if meshFormat==4 then
		extension="stl"
	end
	if sim.getInt32Parameter(sim.intparam_platform)==0 then
		local expDir=string.gsub(exportDir, "/", "\\")
		os.execute("rmdir /S /Q "..'"'..expDir..'"')
		os.execute("mkdir "..'"'..expDir..'"')
	else
		local expDir=string.gsub(exportDir, " ", "\\ ")
		os.execute("rm -rf "..expDir)
		os.execute("mkdir "..expDir)
	end

	local file=io.open(exportDir.."/"..fileName,"w")

	file:write("// Data format:\n")
	file:write("//\n")
	file:write("// For each scene object:\n")
	file:write("//    name{<SCENE_OBJECT_NAME>} ref{<ABS_OBJECT_REFERENCE_FRAME_MATRIX>} parent{PARENT_OBJECT_NAME} visibility{<0_OR_1>} type{<OBJECT_TYPE>} \n")
	file:write("//\n")
	file:write("//\n")
	file:write("// Where:\n")
	file:write("//  <ABS_OBJECT_REFERENCE_FRAME_MATRIX> is: Xx Yx Zx POSx Xy Yy Zy POSy Xz Yz Zz POSz 0 0 0 1\n")
	file:write("//  <PARENT_OBJECT_NAME> is: * if the object has no parent\n")
	file:write("//  <OBJECT_TYPE> is: object, joint, shape or multishape\n")
	file:write("//\n")
	file:write("//\n")
	file:write("// If <OBJECT_TYPE> is shape, then following line describes the shape mesh.\n")
	file:write("// If <OBJECT_TYPE> is multishape, then following lines describe the shape mesh components.\n")
	file:write("//\n")
	file:write("//\n")
	file:write("// A mesh is described with:\n")
	file:write("//  file{<FILE_NAME>} color{<MESH_COLOR>}\n")
	file:write("//\n")
	file:write("//\n")
	file:write("// Where:\n")
	file:write("//  <MESH_COLOR> is: ambientR ambientG ambientB specularR specularG specularB\n")
	file:write("//\n")
	file:write("//\n")
	file:write("// If <OBJECT_TYPE> is joint, then following line describes the joint.\n")
	file:write("//\n")
	file:write("//\n")
	file:write("// A joint is described with:\n")
	file:write("//  type{<JOINT_TYPE>} position{<JOINT_POSITION>} limits{<JOINT_LIMITS>}\n")
	file:write("//\n")
	file:write("//\n")
	file:write("// Where:\n")
	file:write("//  <JOINT_TYPE> is prismatic, revolute or spherical.\n")
	file:write("//  <JOINT_POSITION> is the linear or angular position, or the intrinsic transformation matrix of the spherical joint\n")
	file:write("//  <JOINT_LIMITS> is limitLow limitHigh (for prismatic or revolute joint), cyclic (for revolute joints), none (for spherical joints).\n")
	file:write("//\n")
	file:write("//\n")

	local selectedObjects=sim.getObjectSelection()
	local allObjects=sim.getObjectsInTree(sim.handle_scene)
	if selectedObjects and #selectedObjects==1 then
		allObjects=sim.getObjectsInTree(selectedObjects[1])
	end
	local allIndividualShapesToRemove={}
	local visibleLayers=sim.getInt32Parameter(sim.intparam_visible_layers)
	for obji=1,#allObjects,1 do
		local objType=sim.getObjectType(allObjects[obji])
		local objName=sim.getObjectName(allObjects[obji])
		local matr=sim.getObjectMatrix(allObjects[obji],-1)
		local parentName="*"
		local parentHandle=sim.getObjectParent(allObjects[obji])
		if parentHandle~=-1 then
			parentName=sim.getObjectName(parentHandle)
		end
		local res,layers=sim.getObjectInt32Parameter(allObjects[obji],10)
		file:write("name{"..objName.."}")
		file:write(" ref{")
		for i=1,12,1 do
			file:write(string.format("%e ",matr[i]))
		end
		file:write(string.format("%e %e %e %e}",0,0,0,1))
		file:write(" parent{"..parentName.."}")
		file:write(" visibility{"..sim.boolAnd32(visibleLayers,layers).."}")
		file:write(" type{")
		if objType==sim.object_shape_type then
			local res,param=sim.getObjectInt32Parameter(allObjects[obji],3016)
			if exportIndividualShapeComponents and (param~=0) then
				file:write("multishape}")
				local tobj=sim.copyPasteObjects({allObjects[obji]},0)
				local individualShapes=sim.ungroupShape(tobj[1])
				for j=1,#individualShapes,1 do
					allIndividualShapesToRemove[#allIndividualShapesToRemove+1]=individualShapes[j]
					local indivName=sim.getObjectName(individualShapes[j])
					local indivMatr=sim.getObjectMatrix(individualShapes[j],-1)
                    local totIndivMatrix=sim.getObjectMatrix(allObjects[obji],-1)
					sim.invertMatrix(totIndivMatrix)
					totIndivMatrix=sim.multiplyMatrices(totIndivMatrix,indivMatr)
					local vertices,indices=sim.getShapeMesh(individualShapes[j])
					file:write("\n    file{"..indivName.."."..extension.."}")
					local result,col1=sim.getShapeColor(individualShapes[j],nil,sim.colorcomponent_ambient_diffuse)
					local result,col2=sim.getShapeColor(individualShapes[j],nil,sim.colorcomponent_specular)
					file:write(string.format(" color{%.2f %.2f %.2f %.2f %.2f %.2f}",col1[1],col1[2],col1[3],col2[1],col2[2],col2[3]))
					if vertices then
						for i=1,#vertices/3,1 do
							local v={vertices[3*(i-1)+1],vertices[3*(i-1)+2],vertices[3*(i-1)+3]}
							v=sim.multiplyVector(totIndivMatrix,v)
							vertices[3*(i-1)+1]=v[1]
							vertices[3*(i-1)+2]=v[2]
							vertices[3*(i-1)+3]=v[3]
						end
						sim.exportMesh(meshFormat,exportDir.."/"..indivName.."."..extension,0,1,{vertices},{indices},nil,{indivName})
					end
				end
			else
				file:write("shape}")
				file:write("\n    file{"..objName.."."..extension.."}")
				local result,col1=sim.getShapeColor(allObjects[obji],nil,sim.colorcomponent_ambient_diffuse)
				local result,col2=sim.getShapeColor(allObjects[obji],nil,sim.colorcomponent_specular)
				file:write(string.format(" color{%.2f %.2f %.2f %.2f %.2f %.2f}",col1[1],col1[2],col1[3],col2[1],col2[2],col2[3]))
				local vertices,indices=sim.getShapeMesh(allObjects[obji])
				if vertices then
					sim.exportMesh(meshFormat,exportDir.."/"..objName.."."..extension,0,1,{vertices},{indices},nil,{objName})
				end
			end
		else
			if objType==sim.object_joint_type then
				file:write("joint}")
				local t=sim.getJointType(allObjects[obji])
				local cyclic,interval=sim.getJointInterval(allObjects[obji])
				if t==sim.joint_prismatic_subtype then
					local pos=sim.getJointPosition(allObjects[obji])
					file:write(string.format("\n    type{prismatic} position{%e} limits{%e %e}",pos,interval[1],interval[1]+interval[2]))
				end
				if t==sim.joint_revolute_subtype then
					local pos=sim.getJointPosition(allObjects[obji])
					if cyclic then
						file:write(string.format("\n    type{revolute} position{%e} limits{cyclic}",pos))
					else
						file:write(string.format("\n    type{revolute} position{%e} limits{%e %e}",pos,interval[1],interval[1]+interval[2]))
					end
				end
				if t==sim.joint_spherical_subtype then
					local jmatr=sim.getJointMatrix(allObjects[obji])
					file:write("\n    type{spherical} position{")
					for i=1,12,1 do
						file:write(string.format("%e ",jmatr[i]))
					end
					file:write(string.format("%e %e %e %e} limits{none}",0,0,0,1))
				end
			else
				file:write("object}")
			end
		end
		file:write("\n")
	end
	file:close()
	for i=1,#allIndividualShapesToRemove,1 do
		sim.removeObject(allIndividualShapesToRemove[i])
	end
end