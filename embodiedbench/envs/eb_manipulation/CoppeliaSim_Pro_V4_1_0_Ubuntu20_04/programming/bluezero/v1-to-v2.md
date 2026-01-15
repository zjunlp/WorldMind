# Porting to BlueZero v2

## b0::init

Now it is mandatory to call `b0::init(int, char**)` prior to node constructions, in order to process command line arguments.

## error: ambiguous call to XXX

Where XXX can be:

 - b0::Publisher::Publisher(...)
 - b0::Subscriber::Subscriber(...)
 - b0::ServiceClient::ServiceClient(...)
 - b0::ServiceServer::ServiceServer(...)

Solution: cast the `callback` argument to the correct type, choosing one from:

 - CallbackRaw
 - CallbackRawType
 - CallbackParts
 - CallbackMsg<T>
 - CallbackMsgParts<T>

Example: `static_cast<b0::Subscriber::CallbackRaw>(callback)`.

Note: for specifying no callback (i.e. empty callback) there is an overload of Subscriber and ServiceServer constructors without the callback argument.

## Other API changes

`b0::Node::spin()` now accepts a callback method as first argument.
