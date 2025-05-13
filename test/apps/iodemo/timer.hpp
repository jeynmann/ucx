#include <stdint.h>
#include <chrono>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

namespace timer {

class Timer {
public:
    using Tp = std::chrono::time_point<std::chrono::system_clock>;

    inline constexpr Timer() {}

    inline void start() {
        __s = std::chrono::system_clock::now();
    }

    inline void stop() {
        __e = std::chrono::system_clock::now();
        __d = std::chrono::duration_cast<std::chrono::nanoseconds>(__e - __s).count();
    }

    inline explicit operator uint64_t() const {
        return __d;
    }

    inline void operator()() {
        __s = std::move(__e);
        __e = std::chrono::system_clock::now();
        __d = std::chrono::duration_cast<std::chrono::nanoseconds>(__e - __s).count();
    }

    inline bool operator<(const Timer& other) const {
        return operator uint64_t() < other.__d;
    }

private:
    Tp __s{}, __e{};
    uint64_t __d{};

};

class CTimer {
public:
    using Tp = struct timeval;

    inline CTimer() {}

    inline void start() {
        gettimeofday(&__s, NULL);
    }

    inline void stop() {
        gettimeofday(&__e, NULL);
        __d  = __e.tv_usec - __s.tv_usec;
        // __ds = (__e.tv_sec - __s.tv_sec) * 1000000 + __d;
    }

    inline explicit operator uint64_t() const {
        return  __d;
    }

    inline void operator()() {
        __s = __e;
        gettimeofday(&__e, NULL);
        __d = __e.tv_usec - __s.tv_usec;
        // __ds = (__e.tv_sec - __s.tv_sec) * 1000000 + __d;
    }

    inline bool operator<(const CTimer& other) const {
        return operator uint64_t() < other.__d;
    }

private:
    Tp __s;
    Tp __e;
    uint64_t __d;
    // uint64_t __ds;

};

class Collector {
public:
    using Timer = CTimer;
    inline Collector() : __h(""){}
    inline Collector(const char* name) : __h(name){}

    Collector(const Collector&) = default;
    Collector(Collector&&) = default;
    Collector& operator=(const Collector&) = default;
    Collector& operator=(Collector&&) = default;

    inline void rename(const char* name) {
        __h =  name;
    }

    inline void resize(size_t n) {
        __t.resize(n);
    }

    inline void reserve(size_t n) {
        __t.reserve(n);
    }

    inline size_t size() const {
        return __t.size();
    }

    inline explicit operator uint64_t() const {
        const size_t n = __t.size();
        size_t sum = 0;
        for (size_t i = 0; i != n; ++i) {
            sum += (uint64_t)__t[i];
        }
        return sum / n;
    }

    inline Timer& operator[](size_t i) {
        return __t[i];
    }

    inline std::stringstream& stream() const {
        return __ss;
    }

    inline std::stringstream& to_stream() const {
        __to_stream();
        return __ss;
    }

    inline std::string to_string() const {
        __to_stream();
        return __ss.str();
    }

    inline std::string to_string_sorted() {
        std::sort(__t.begin(), __t.end());
        return to_string();
    }

private:
    void __to_stream() const {
        const size_t n = __t.size();
        __ss << "<" << __h << "> ";
        if (n != 1) {
            __ss << "cnt:" << n << " ";
            __ss << "avg:" << operator uint64_t() << "us ";
            __ss << "all:";
        }
        for (size_t i = 0; i != n; ++i) {
            __ss << (uint64_t) __t[i] << "us ";
        }
    }

    const char* __h = "";
    std::vector<Timer> __t;
    mutable std::stringstream __ss;
};

class Factory {
public:
    static inline Factory& inst() {
        static Factory _;
        return _;
    }

    class Proxy {
    public:
        inline constexpr Proxy() {}
        inline constexpr Proxy(Collector* p) : __p(p) {}
        inline Proxy(Proxy&& other) : __p(other.__p) {
            other.__p = nullptr;
        }
        inline Proxy& operator=(Proxy&& other) {
            if (__p != nullptr) {
                Factory::inst().put(__p);
            }
            __p = other.__p;
            other.__p = nullptr;
            return *this;
        }
        inline ~Proxy() {
            Factory::inst().put(__p);
        }
        inline explicit operator bool() const {
            return __p == nullptr;
        }
        inline explicit operator uint64_t() const {
            return __p->operator uint64_t();
        }
        inline Collector::Timer& operator[](size_t i) const {
            return __p->operator[](i);
        }
        inline Collector* operator->() const {
            return __p;
        }
        inline Collector& operator*() const {
            return *__p;
        }
    private:
        Collector* __p = nullptr;
    };

    inline Proxy pop() {
        return pop_raw();
    }

    inline Proxy pop_named(const char* name) {
        auto* ptr = pop_raw();
        ptr->rename(name);
        return ptr;
    }

    inline Collector* pop_raw() {
        if (__i == 0) {
            __add(__n >> 1);
        }
        return __o[--__i];
    }

    inline void put(Collector* p) {
        __o[__i++] = p;
    }

private:
    inline Factory() {
        __add(64);
    }

    Factory(const Factory&) = delete;
    Factory& operator=(const Factory&) = delete;

    void __add(size_t n) {
        __i = n;
        __n += n;
        __o.resize(__n);
        for(size_t i = 0; i != n; ++i) {
            __o[i] = new Collector;
            __o[i]->resize(1);
        }
    }

    size_t __i = 0;
    size_t __n = 0;
    std::vector<Collector*> __o;
};

struct SystemTap {
    static inline SystemTap& global() {
        static SystemTap __inst;
        return __inst;
    }
    void beg_stap() __attribute__((noinline));
    void end_stap() __attribute__((noinline));
    struct timeval stap_ts;
    struct timeval stap_te;
};

__attribute__((weak)) void SystemTap::beg_stap() {
    gettimeofday(&stap_ts, NULL);
}

__attribute__((weak)) void SystemTap::end_stap() {
    gettimeofday(&stap_te, NULL);
    printf("<system-tap> %ldus\n", (stap_te.tv_usec - stap_ts.tv_usec));
}

}

#if 0
extern "C" {
    extern void* get_timer(const char* name);
    extern void start_timer(void* timer);
    extern void end_timer(void* timer);
    extern void rel_timer(void* timer);

    void* get_timer(const char* name) {
        auto p = timer::Factory::inst().pop_raw();
        p->rename(name);
        return p;
    }

    void start_timer(void* timer) {
        reinterpret_cast<timer::Collector*>(timer)->operator[](0)();
    }

    void end_timer(void* timer) {
        reinterpret_cast<timer::Collector*>(timer)->operator[](0)();
    }

    void rel_timer(void* timer) {
        auto p = reinterpret_cast<timer::Collector*>(timer);
        std::cout << std::move(p->to_string());
        timer::Factory::inst().put(p);
    }
}

    struct timeval ts;
    struct timeval te;
    gettimeofday(&ts, NULL);
    gettimeofday(&te, NULL);
    ucs_warn("<func> used %ldus\n", (te.tv_usec - ts.tv_usec));
#endif
