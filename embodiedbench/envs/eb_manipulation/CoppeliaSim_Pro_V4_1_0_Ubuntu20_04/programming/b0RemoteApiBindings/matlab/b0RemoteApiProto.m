function [methodinfo,structs,enuminfo,ThunkLibName]=b0RemoteApiProto

ival={cell(1,0)}; 
structs=[];enuminfo=[];fcnNum=1;
fcns=struct('name',ival,'calltype',ival,'LHS',ival,'RHS',ival,'alias',ival);
ThunkLibName=[];

fcns.name{fcnNum}='b0_is_initialized'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='int32'; fcns.RHS{fcnNum}=[];fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_init'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='int32'; fcns.RHS{fcnNum}={'int32Ptr','stringPtrPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_buffer_new'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='voidPtr'; fcns.RHS{fcnNum}={'uint64'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_buffer_delete'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;

fcns.name{fcnNum}='b0_node_new'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='voidPtr'; fcns.RHS{fcnNum}={'int8Ptr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_node_delete'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_node_init'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_node_spin_once'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_node_spin'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_node_time_usec'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='long'; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_node_hardware_time_usec'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='long'; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;

fcns.name{fcnNum}='b0_service_client_new_ex'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='voidPtr'; fcns.RHS{fcnNum}={'voidPtr','int8Ptr','int32','int32'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_service_client_delete'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_service_client_call'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='uint8Ptr'; fcns.RHS{fcnNum}={'voidPtr','uint8Ptr','uint64','uint64Ptr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_service_client_set_option'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='int32'; fcns.RHS{fcnNum}={'voidPtr','int32','int32'};fcnNum=fcnNum+1;

fcns.name{fcnNum}='b0_publisher_new_ex'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='voidPtr'; fcns.RHS{fcnNum}={'voidPtr','int8Ptr','int32','int32'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_publisher_delete'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_publisher_init'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_publisher_publish'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr','int8Ptr','uint64'};fcnNum=fcnNum+1;

fcns.name{fcnNum}='b0_subscriber_new_ex'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='voidPtr'; fcns.RHS{fcnNum}={'voidPtr','int8Ptr','voidPtr','int32','int32'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_subscriber_delete'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_subscriber_init'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}=[]; fcns.RHS{fcnNum}={'voidPtr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_subscriber_poll'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='int32'; fcns.RHS{fcnNum}={'voidPtr','long'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_subscriber_read'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='uint8Ptr'; fcns.RHS{fcnNum}={'voidPtr','uint64Ptr'};fcnNum=fcnNum+1;
fcns.name{fcnNum}='b0_subscriber_set_option'; fcns.calltype{fcnNum}='cdecl'; fcns.LHS{fcnNum}='int32'; fcns.RHS{fcnNum}={'voidPtr','int32','int32'};fcnNum=fcnNum+1;

methodinfo=fcns;