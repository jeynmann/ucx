/**
 * Copyright (c) NVIDIA CORPORATION & AFFILIATES, 2020. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "proto_single.h"
#include "proto_common.h"
#include "proto_init.h"
#include "proto_debug.h"

#include <ucs/debug/assert.h>
#include <ucs/debug/log.h>
#include <ucs/sys/math.h>


ucs_status_t ucp_proto_single_init(const ucp_proto_single_init_params_t *params,
                                   ucp_proto_perf_t **perf_p,
                                   ucp_proto_single_priv_t *spriv)
{
    const char *proto_name = ucp_proto_id_field(params->super.super.proto_id,
                                                name);
    ucp_proto_perf_node_t *tl_perf_node;
    ucp_proto_common_tl_perf_t tl_perf;
    ucp_lane_index_t num_lanes;
    ucp_md_map_t reg_md_map;
    ucp_lane_index_t lane;
    ucs_status_t status;

    if (!ucp_proto_common_check_memtype_copy(&params->super)) {
        return UCS_ERR_UNSUPPORTED;
    }

    num_lanes = ucp_proto_common_find_lanes(
            &params->super.super, params->super.flags, params->lane_type,
            params->tl_cap_flags, 1, params->super.exclude_map,
            ucp_proto_common_filter_min_frag, &lane);
    if (num_lanes == 0) {
        ucs_trace("no lanes for %s",
                  ucp_proto_id_field(params->super.super.proto_id, name));
        return UCS_ERR_NO_ELEM;
    }

    ucs_assert(num_lanes == 1);

    reg_md_map = ucp_proto_common_reg_md_map(&params->super, UCS_BIT(lane));
    if (reg_md_map == 0) {
        spriv->reg_md = UCP_NULL_RESOURCE;
    } else {
        ucs_assert(ucs_popcount(reg_md_map) == 1);
        spriv->reg_md = ucs_ffs64(reg_md_map);
    }

    ucp_proto_common_lane_priv_init(&params->super, reg_md_map, lane,
                                    &spriv->super);
    status = ucp_proto_common_get_lane_perf(&params->super, lane, &tl_perf,
                                            &tl_perf_node);
    if (status != UCS_OK) {
        return status;
    }

    status = ucp_proto_init_perf(&params->super, &tl_perf, tl_perf_node,
                                 reg_md_map, proto_name, perf_p);
    ucp_proto_perf_node_deref(&tl_perf_node);

    return status;
}

void ucp_proto_single_probe(const ucp_proto_single_init_params_t *params)
{
    ucp_proto_single_priv_t spriv;
    ucp_proto_perf_t *perf;
    ucs_status_t status;

    if (!ucp_proto_common_init_check_err_handling(&params->super)) {
        return;
    }

    status = ucp_proto_single_init(params, &perf, &spriv);
    if (status != UCS_OK) {
        return;
    }

    ucp_proto_select_add_proto(&params->super.super, params->super.cfg_thresh,
                               params->super.cfg_priority, perf, &spriv,
                               sizeof(spriv));
}

void ucp_proto_single_query(const ucp_proto_query_params_t *params,
                            ucp_proto_query_attr_t *attr)
{
    UCS_STRING_BUFFER_FIXED(config_strb, attr->config, sizeof(attr->config));
    const ucp_proto_single_priv_t *spriv = params->priv;

    ucp_proto_default_query(params, attr);
    ucp_proto_common_lane_priv_str(params, &spriv->super, 1, 1, &config_strb);
    attr->lane_map = UCS_BIT(spriv->super.lane);
}
