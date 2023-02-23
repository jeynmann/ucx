/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2001-2018. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCS_PROFILE_OFF_H_
#define UCS_PROFILE_OFF_H_

#include "profile_defs.h"

BEGIN_C_DECLS

#define UCS_PROFILE_SAMPLE(...)                             UCS_EMPTY_STATEMENT
#define UCS_PROFILE_CODE(_, _code)                          _code
#define UCS_PROFILE_FUNC(_ret_type, _name, _, ...)          _ret_type _name(__VA_ARGS__)
#define UCS_PROFILE_FUNC_VOID(_name, _, ...)                void _name(__VA_ARGS__)
#define UCS_PROFILE_NAMED_CALL(_name, _func, ...)           _func(__VA_ARGS__)
#define UCS_PROFILE_CALL(_func, ...)                        _func(__VA_ARGS__)
#define UCS_PROFILE_NAMED_CALL_VOID                         UCS_PROFILE_NAMED_CALL
#define UCS_PROFILE_CALL_VOID                               UCS_PROFILE_CALL
#define UCS_PROFILE_REQUEST_NEW(...)                        UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_EVENT(...)                      UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_EVENT_CHECK_STATUS(...)         UCS_EMPTY_STATEMENT
#define UCS_PROFILE_REQUEST_FREE(...)                       UCS_EMPTY_STATEMENT
#define UCS_PROFILE_CLASS_INIT_FUNC(_type, arglist, ...)    UCS_CLASS_INIT_FUNC(_type, ## __VA_ARGS__)
#define UCS_PROFILE_CLASS_CLEANUP_FUNC                      UCS_CLASS_CLEANUP_FUNC
#define UCS_PROFILE_CLASS_DEFINE_NAMED_DELETE_FUNC          UCS_CLASS_DEFINE_NAMED_DELETE_FUNC
#define UCS_PROFILE_CLASS_DEFINE_DELETE_FUNC                UCS_CLASS_DEFINE_DELETE_FUNC

END_C_DECLS

#endif
