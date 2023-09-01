/*
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2019. ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include <sys/epoll.h>
#include "unistd.h"
#include "errno.h"
#include "jucx_common_def.h"
#include "org_openucx_jucx_Epoll.h"

JNIEXPORT jlong JNICALL
Java_org_openucx_jucx_Epoll_allocEpollEventsNative(JNIEnv *env, jclass cls, jint size) {
    return (jlong)new epoll_event[size];
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_Epoll_freeEpollEventsNative(JNIEnv *env, jclass cls, jlong ptr) {
    delete[] (epoll_event*)ptr;
}

JNIEXPORT jint JNICALL
Java_org_openucx_jucx_Epoll_getEpollEventsOpsNative(JNIEnv *env, jclass cls, jlong ptr, jint id) {
    epoll_event* ep_ptr = (epoll_event*)ptr;
    return ep_ptr[id].events;
}

JNIEXPORT jint JNICALL
Java_org_openucx_jucx_Epoll_getEpollEventsFdNative(JNIEnv *env, jclass cls, jlong ptr, jint id) {
    epoll_event* ep_ptr = (epoll_event*)ptr;
    return ep_ptr[id].data.fd;
}

JNIEXPORT jint JNICALL
Java_org_openucx_jucx_Epoll_epollCreateNative(JNIEnv *env, jclass cls, jint size)
{
    return epoll_create(size);
}

JNIEXPORT void JNICALL
Java_org_openucx_jucx_Epoll_epollCloseNative(JNIEnv *env, jclass cls, jint epfd)
{
    close(epfd);
}

JNIEXPORT jint JNICALL
Java_org_openucx_jucx_Epoll_epollCtlNative(JNIEnv *env, jclass cls, jint epfd, jint op, jint fd)
{
    epoll_event event;
    event.events = op;
    event.data.fd = fd;
    return epoll_ctl(epfd, op, fd, &event);
}

JNIEXPORT jint JNICALL
Java_org_openucx_jucx_Epoll_epollWaitNative(JNIEnv *env, jclass cls, jint epfd, jlong events,
		       jint maxevents, jint timeout)
{
    return epoll_wait(epfd, (epoll_event *)events, maxevents, timeout);
}

// JNIEXPORT jint JNICALL
// Java_org_openucx_jucx_Epoll_epollAddFdNative(JNIEnv *env, jclass cls, jint epfd, jint fd)
// {
//     epoll_event ev;
//     ev.events = EPOLLIN;
//     ev.data.fd = fd;

//     return epoll_ctl(epfd, EPOLL_CTL_ADD, fd, &ev);
// }

// JNIEXPORT jint JNICALL
// Java_org_openucx_jucx_Epoll_epollWaitFdNative(JNIEnv *env, jclass cls, jint epfd)
// {
//     epoll_event ev;

//     jint ret;
//     do {
//          ret = epoll_wait(epfd, &ev, 1, -1);
//     } while ((ret == -1) && (errno == EINTR || errno == EAGAIN));

//     return ev.data.fd;
// }

JNIEXPORT void JNICALL 
Java_org_openucx_jucx_Epoll_loadNative(JNIEnv *env, jclass cls)
{
    JUCX_DEFINE_INT_CONSTANT(EPOLLIN);
    JUCX_DEFINE_INT_CONSTANT(EPOLLPRI);
    JUCX_DEFINE_INT_CONSTANT(EPOLLOUT);
    JUCX_DEFINE_INT_CONSTANT(EPOLLRDNORM);
    JUCX_DEFINE_INT_CONSTANT(EPOLLRDBAND);
    JUCX_DEFINE_INT_CONSTANT(EPOLLWRNORM);
    JUCX_DEFINE_INT_CONSTANT(EPOLLWRBAND);
    JUCX_DEFINE_INT_CONSTANT(EPOLLMSG);
    JUCX_DEFINE_INT_CONSTANT(EPOLLERR);
    JUCX_DEFINE_INT_CONSTANT(EPOLLHUP);
    JUCX_DEFINE_INT_CONSTANT(EPOLLRDHUP);
    JUCX_DEFINE_INT_CONSTANT(EPOLLWAKEUP);
    JUCX_DEFINE_INT_CONSTANT(EPOLLONESHOT);
    JUCX_DEFINE_INT_CONSTANT(EPOLLET);

    JUCX_DEFINE_INT_CONSTANT(EPOLL_CTL_ADD);
    JUCX_DEFINE_INT_CONSTANT(EPOLL_CTL_DEL);
    JUCX_DEFINE_INT_CONSTANT(EPOLL_CTL_MOD);

    // JUCX_DEFINE_INT_CONSTANT(EINTR);
    // JUCX_DEFINE_INT_CONSTANT(EAGAIN);
}
