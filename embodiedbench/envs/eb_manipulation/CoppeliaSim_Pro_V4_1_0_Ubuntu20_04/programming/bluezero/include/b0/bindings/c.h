#ifndef B0__C_H__INCLUDED
#define B0__C_H__INCLUDED

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <inttypes.h>

// logger level:
#define B0_FATAL 600
#define B0_ERROR 500
#define B0_WARN  400
#define B0_INFO  300
#define B0_DEBUG 200
#define B0_TRACE 100

// docket options:
#define B0_SOCK_OPT_LINGERPERIOD   1
#define B0_SOCK_OPT_BACKLOG        2
#define B0_SOCK_OPT_READTIMEOUT    3
#define B0_SOCK_OPT_WRITETIMEOUT   4
#define B0_SOCK_OPT_IMMEDIATE      5
#define B0_SOCK_OPT_CONFLATE       6
#define B0_SOCK_OPT_READHWM        7
#define B0_SOCK_OPT_WRITEHWM       8

#ifndef B0_EXPORT
#ifdef _WIN32
#ifdef B0_LIBRARY
#define B0_EXPORT __declspec(dllexport)
#else // B0_LIBRARY
#define B0_EXPORT __declspec(dllimport)
#endif // B0_LIBRARY
#else // _WIN32
#define B0_EXPORT
#endif // _WIN32
#endif // B0_EXPORT

struct b0_node;
typedef struct b0_node b0_node;

struct b0_publisher;
typedef struct b0_publisher b0_publisher;

struct b0_subscriber;
typedef struct b0_subscriber b0_subscriber;

struct b0_service_client;
typedef struct b0_service_client b0_service_client;

struct b0_service_server;
typedef struct b0_service_server b0_service_server;

B0_EXPORT int b0_init(int *argc, char **argv);
B0_EXPORT int b0_is_initialized();
B0_EXPORT int b0_add_option(const char *name, const char *descr);
B0_EXPORT int b0_add_option_string(const char *name, const char *descr, int required, const char *def);
B0_EXPORT int b0_add_option_int(const char *name, const char *descr, int required, int def);
B0_EXPORT int b0_add_option_int64(const char *name, const char *descr, int required, int64_t def);
B0_EXPORT int b0_add_option_double(const char *name, const char *descr, int required, double def);
B0_EXPORT int b0_add_option_string_vector(const char *name, const char *descr, int required, const char **def, int def_count);
B0_EXPORT int b0_add_option_int_vector(const char *name, const char *descr, int required, int *def, int def_count);
B0_EXPORT int b0_add_option_int64_vector(const char *name, const char *descr, int required, int64_t *def, int def_count);
B0_EXPORT int b0_add_option_double_vector(const char *name, const char *descr, int required, double *def, int def_count);
B0_EXPORT int b0_set_positional_option(const char *name, int max_count);
B0_EXPORT int b0_has_option(const char *name);
B0_EXPORT int b0_get_option_string(const char *name, char **out);
B0_EXPORT int b0_get_option_int(const char *name, int *out);
B0_EXPORT int b0_get_option_int64(const char *name, int64_t *out);
B0_EXPORT int b0_get_option_double(const char *name, double *out);
B0_EXPORT int b0_get_option_string_vector(const char *name, char ***out, int *count);
B0_EXPORT int b0_get_option_int_vector(const char *name, int **out, int *count);
B0_EXPORT int b0_get_option_int64_vector(const char *name, int64_t **out, int *count);
B0_EXPORT int b0_get_option_double_vector(const char *name, double **out, int *count);
B0_EXPORT int b0_get_console_log_level();
B0_EXPORT int b0_set_console_log_level(int level);
B0_EXPORT double b0_get_spin_rate();
B0_EXPORT int b0_set_spin_rate(double rate);
B0_EXPORT int b0_quit_requested();
B0_EXPORT int b0_quit();

B0_EXPORT void * b0_buffer_new(size_t size);
B0_EXPORT void b0_buffer_delete(void *buffer);

B0_EXPORT b0_node * b0_node_new(const char *name);
B0_EXPORT void b0_node_delete(b0_node *node);
B0_EXPORT int b0_node_init(b0_node *node);
B0_EXPORT int b0_node_shutdown(b0_node *node);
B0_EXPORT int b0_node_shutdown_requested(b0_node *node);
B0_EXPORT int b0_node_spin_once(b0_node *node);
B0_EXPORT int b0_node_spin(b0_node *node);
B0_EXPORT int b0_node_cleanup(b0_node *node);
B0_EXPORT const char * b0_node_get_name(b0_node *node);
B0_EXPORT int b0_node_get_state(b0_node *node);
B0_EXPORT void * b0_node_get_context(b0_node *node);
B0_EXPORT int64_t b0_node_hardware_time_usec(b0_node *node);
B0_EXPORT int64_t b0_node_time_usec(b0_node *node);
B0_EXPORT void b0_node_sleep_usec(b0_node *node, int64_t usec);
B0_EXPORT int b0_node_log(b0_node *node, int level, const char *message);

B0_EXPORT b0_publisher * b0_publisher_new_ex(b0_node *node, const char *topic_name, int managed, int notify_graph);
B0_EXPORT b0_publisher * b0_publisher_new(b0_node *node, const char *topic_name);
B0_EXPORT void b0_publisher_delete(b0_publisher *pub);
B0_EXPORT int b0_publisher_init(b0_publisher *pub);
B0_EXPORT int b0_publisher_cleanup(b0_publisher *pub);
B0_EXPORT int b0_publisher_spin_once(b0_publisher *pub);
B0_EXPORT const char * b0_publisher_get_topic_name(b0_publisher *pub);
B0_EXPORT int b0_publisher_publish(b0_publisher *pub, const void *data, size_t size);
B0_EXPORT int b0_publisher_log(b0_publisher *pub, int level, const char *message);
B0_EXPORT int b0_publisher_set_option(b0_publisher *pub, int option, int value);

B0_EXPORT b0_subscriber * b0_subscriber_new_ex(b0_node *node, const char *topic_name, void (*callback)(const void *, size_t), int managed, int notify_graph);
B0_EXPORT b0_subscriber * b0_subscriber_new(b0_node *node, const char *topic_name, void (*callback)(const void *, size_t));
B0_EXPORT void b0_subscriber_delete(b0_subscriber *sub);
B0_EXPORT int b0_subscriber_init(b0_subscriber *sub);
B0_EXPORT int b0_subscriber_cleanup(b0_subscriber *sub);
B0_EXPORT int b0_subscriber_spin_once(b0_subscriber *sub);
B0_EXPORT const char * b0_subscriber_get_topic_name(b0_subscriber *sub);
B0_EXPORT int b0_subscriber_log(b0_subscriber *sub, int level, const char *message);
B0_EXPORT int b0_subscriber_poll(b0_subscriber *sub, long timeout);
B0_EXPORT void * b0_subscriber_read(b0_subscriber *sub, size_t *size);
B0_EXPORT int b0_subscriber_set_option(b0_subscriber *sub, int option, int value);

B0_EXPORT b0_service_client * b0_service_client_new_ex(b0_node *node, const char *service_name, int managed, int notify_graph);
B0_EXPORT b0_service_client * b0_service_client_new(b0_node *node, const char *service_name);
B0_EXPORT void b0_service_client_delete(b0_service_client *cli);
B0_EXPORT int b0_service_client_init(b0_service_client *cli);
B0_EXPORT int b0_service_client_cleanup(b0_service_client *cli);
B0_EXPORT int b0_service_client_spin_once(b0_service_client *cli);
B0_EXPORT const char * b0_service_client_get_service_name(b0_service_client *cli);
B0_EXPORT void * b0_service_client_call(b0_service_client *cli, const void *data, size_t size, size_t *out_size);
B0_EXPORT int b0_service_client_log(b0_service_client *cli, int level, const char *message);
B0_EXPORT int b0_service_client_set_option(b0_service_client *cli, int option, int value);

B0_EXPORT b0_service_server * b0_service_server_new_ex(b0_node *node, const char *service_name, void * (*callback)(const void *, size_t, size_t *), int managed, int notify_graph);
B0_EXPORT b0_service_server * b0_service_server_new(b0_node *node, const char *service_name, void * (*callback)(const void *, size_t, size_t *));
B0_EXPORT void b0_service_server_delete(b0_service_server *srv);
B0_EXPORT int b0_service_server_init(b0_service_server *srv);
B0_EXPORT int b0_service_server_cleanup(b0_service_server *srv);
B0_EXPORT int b0_service_server_spin_once(b0_service_server *srv);
B0_EXPORT const char * b0_service_server_get_service_name(b0_service_server *srv);
B0_EXPORT int b0_service_server_log(b0_service_server *srv, int level, const char *message);
B0_EXPORT int b0_service_server_poll(b0_service_server *srv, long timeout);
B0_EXPORT void * b0_service_server_read(b0_service_server *srv, size_t *size);
B0_EXPORT int b0_service_server_write(b0_service_server *srv, const void *msg, size_t size);
B0_EXPORT int b0_service_server_set_option(b0_service_server *srv, int option, int value);

#ifdef __cplusplus
}
#endif

#endif // B0__C_H__INCLUDED
