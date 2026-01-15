#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <b0/bindings/c.h>

pid_t resolver_pid = -1;
pid_t server_pid = -1;
const char *service_name = "test_service";
int server_wait = 0;
int expect_failure = 0;
int read_success = -1;

void start_resolver()
{
    resolver_pid = fork();

    if(resolver_pid == -1)
    {
        fprintf(stderr, "error: start_resolver: fork failed\n");
        exit(1);
    }
    else if(resolver_pid > 0)
    {
        sleep(2);
    }
    else
    {
        char *argv[2], *envp[2];
        argv[0] = strdup("b0_resolver");
        argv[1] = NULL;
        envp[0] = strdup("B0_HOST_ID=localhost");
        envp[1] = NULL;
        execve("b0_resolver", argv, envp);
        exit(1); // exec() never returns
    }
}

void kill_resolver(int sig)
{
    if(resolver_pid > 0)
        kill(resolver_pid, SIGKILL);
}

void * server_callback(const void *req, size_t sz, size_t *out_sz)
{
    printf("server: Received: %s\n", (const char*)req);

    printf("server: Waiting %d seconds...\n", server_wait);
    sleep(server_wait);

    const char *repmsg = "hi";
    printf("server: Sending: %s\n", repmsg);

    *out_sz = strlen(repmsg);
    void *rep = b0_buffer_new(*out_sz);
    memcpy(rep, repmsg, *out_sz);
    return rep;
}

void start_server()
{
    server_pid = fork();

    if(server_pid == -1)
    {
        fprintf(stderr, "error: start_server: fork failed\n");
        exit(1);
    }
    else if(server_pid == 0)
    {
        b0_node *server_node = b0_node_new("server");
        b0_service_server *srv = b0_service_server_new(server_node, service_name, &server_callback);
        b0_node_init(server_node);
        b0_node_spin(server_node);
        b0_node_cleanup(server_node);
        b0_service_server_delete(srv);
        b0_node_delete(server_node);
    }
}

void kill_server(int sig)
{
    if(server_pid > 0)
        kill(server_pid, SIGKILL);
}

void start_client()
{
    b0_node *client_node = b0_node_new("client");
    b0_service_client *cli = b0_service_client_new(client_node, service_name);
    b0_service_client_set_option(cli, B0_SOCK_OPT_READTIMEOUT, 2000);
    b0_node_init(client_node);

    const char *req = "hello";
    printf("client: Sending: %s\n", req);
    char *rep;
    size_t rep_sz;
    rep = b0_service_client_call(cli, req, strlen(req) + 1, &rep_sz);
    read_success = rep ? 1 : 0;
    if(rep)
    {
        printf("client: Received: %s\n", rep);
        b0_buffer_delete(rep);
    }
    else
    {
        printf("client: Service call failed (timeout?)\n");
    }

    b0_node_cleanup(client_node);
    b0_service_client_delete(cli);
    b0_node_delete(client_node);
}

void kill_all(int sig)
{
    kill_server(sig);
    kill_resolver(sig);
}

int main(int argc, char **argv)
{
    signal(SIGTERM, (void (*)(int))kill_all);
    signal(SIGINT, (void (*)(int))kill_all);
    signal(SIGABRT, (void (*)(int))kill_all);

    b0_add_option_int("server-wait,w", "server wait in seconds", 1, 0);
    b0_add_option_int("expect-failure,f", "test shoudl fail", 1, 0);
    b0_init(&argc, argv);
    b0_get_option_int("server-wait", &server_wait);
    b0_get_option_int("expect-failure", &expect_failure);

    start_resolver();

    start_server();
    sleep(2);

    start_client();

    int status;

    kill_server(SIGTERM);
    waitpid(server_pid, &status, 0);

    kill_resolver(SIGTERM);
    waitpid(resolver_pid, &status, 0);

    if(read_success == -1) return 2;

    return read_success == expect_failure;
}
