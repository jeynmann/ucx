package org.openucx.jucx;

public class Epoll {
    /* EPOLL_EVENTS */
    public static int EPOLLIN;
    public static int EPOLLPRI;
    public static int EPOLLOUT;
    public static int EPOLLRDNORM;
    public static int EPOLLRDBAND;
    public static int EPOLLWRNORM;
    public static int EPOLLWRBAND;
    public static int EPOLLMSG;
    public static int EPOLLERR;
    public static int EPOLLHUP;
    public static int EPOLLRDHUP;
    public static int EPOLLWAKEUP;
    public static int EPOLLONESHOT;
    public static int EPOLLET;
  
    /* Valid opcodes ( "op" parameter ) to issue to epoll_ctl().  */
    public static int EPOLL_CTL_ADD;
    public static int EPOLL_CTL_DEL;
    public static int EPOLL_CTL_MOD;

    static {
        loadNative();
    }

    long epollEvents;
    int epfd;
    int size;

    public Epoll(int size) {
        this.epollEvents = allocEpollEvents(size);
        this.epfd = epollCreate(size);
        this.size = size;
    }

    public void addFD(int fd) {
        epollCtl(this.epfd, Epoll.EPOLL_CTL_ADD, fd);
    }

    public void delFD(int fd) {
        epollCtl(this.epfd, Epoll.EPOLL_CTL_DEL, fd);
    }

    public void await() {
        epollWait(this.epfd, this.epollEvents, this.size, -1);
    }

    public void close() {
        if (this.epfd != 0) {
            epollClose(this.epfd);
            this.epfd = 0;
        }
    }

    /* error number */
    // public static native int EINTR;
    // public static native int EAGAIN;
    public static long allocEpollEvents(int size) {
        return allocEpollEventsNative(size);
    }

    public static void freeEpollEvents(long events) {
        freeEpollEventsNative(events);
    }

    public static int getEpollEventsOps(long events, int id) {
        return getEpollEventsOpsNative(events, id);
    }

    public static int getEpollEventsFd(long events, int id) {
        return getEpollEventsFdNative(events, id);
    }

    public static int epollCreate(int size) {
        return epollCreateNative(size);
    }

    public static void epollClose(int epfd) {
        epollCloseNative(epfd);
    }

    public static int epollWait(int epfd, long events, int maxevents, int timeout) {
        return epollWaitNative(epfd, events, maxevents, timeout);
    }

    public static int epollCtl(int epfd, int op, int fd) {
        return epollCtlNative(epfd, op, fd);
    }

    private static native long allocEpollEventsNative(int size);
    private static native void freeEpollEventsNative(long events);
    private static native int getEpollEventsOpsNative(long events, int id);
    private static native int getEpollEventsFdNative(long events, int id);

    private static native int epollCreateNative(int size);
    private static native void epollCloseNative(int epfd);
    private static native int epollWaitNative(int epfd, long events, int maxevents, int timeout);
    private static native int epollCtlNative(int epfd, int op, int fd);

    // public static native int epollAddFdNative(int epfd, int fd);
    // public static native int epollWaitFdNative(int epfd);

    private static native void loadNative();
}
