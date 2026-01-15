#ifndef B0__MESSAGE__MESSAGE_PART_H__INCLUDED
#define B0__MESSAGE__MESSAGE_PART_H__INCLUDED

#include <string>

#include <b0/b0.h>

namespace b0
{

namespace message
{

/*!
 * \brief A structure to represent a message part
 *
 * \sa MessageEnvelope
 */
struct MessagePart
{
    //! \brief An optional string indicating the type of the payload
    std::string content_type;

    //! \brief Compression algorithm name, or blank if no compression
    std::string compression_algorithm;

    //! \brief Compression level, or 0 if no compression
    int compression_level;

    //! \brief The payload
    std::string payload;
};

} // namespace message

} // namespace b0

#endif // B0__MESSAGE__MESSAGE_PART_H__INCLUDED
