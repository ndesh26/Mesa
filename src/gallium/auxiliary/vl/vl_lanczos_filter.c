/**************************************************************************
 *
 * Copyright 2016 Nayan Deshmukh.
 * All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sub license, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice (including the
 * next paragraph) shall be included in all copies or substantial portions
 * of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
 * IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 **************************************************************************/

#include <stdio.h>

#include "pipe/p_context.h"

#include "tgsi/tgsi_ureg.h"

#include "util/u_draw.h"
#include "util/u_memory.h"
#include "util/u_math.h"
#include "util/u_rect.h"

#include "vl_types.h"
#include "vl_vertex_buffers.h"
#include "vl_lanczos_filter.h"

enum VS_OUTPUT
{
   VS_O_VPOS = 0,
   VS_O_VTEX = 0
};

static void *
create_vert_shader(struct vl_lanczos_filter *filter)
{
   struct ureg_program *shader;
   struct ureg_src i_vpos;
   struct ureg_dst o_vpos, o_vtex;

   shader = ureg_create(PIPE_SHADER_VERTEX);
   if (!shader)
      return NULL;

   i_vpos = ureg_DECL_vs_input(shader, 0);
   o_vpos = ureg_DECL_output(shader, TGSI_SEMANTIC_POSITION, VS_O_VPOS);
   o_vtex = ureg_DECL_output(shader, TGSI_SEMANTIC_GENERIC, VS_O_VTEX);

   ureg_MOV(shader, o_vpos, i_vpos);
   ureg_MOV(shader, o_vtex, i_vpos);

   ureg_END(shader);

   return ureg_create_shader_and_destroy(shader, filter->pipe);
}

static void
create_frag_shader_lanczos(struct ureg_program *shader, struct ureg_src a,
                           struct ureg_src x, struct ureg_dst o_fragment)
{
   struct ureg_dst temp[8];
   unsigned i;

   for(i = 0; i < 8; ++i)
       temp[i] = ureg_DECL_temporary(shader);

   /*
    * temp[0] = (x == 0) ? 1.0f : x
    * temp[7] = (sin(pi * x) * sin ((pi * x)/a)) / x^2
    * o_fragment = (x == 0) ? 1.0f : temp[7]
    */
   ureg_MOV(shader, temp[0], x);
   ureg_SEQ(shader, temp[1], x, ureg_imm1f(shader, 0.0f));

   ureg_LRP(shader, temp[0], ureg_src(temp[1]),
            ureg_imm1f(shader, 1.0f), ureg_src(temp[0]));

   ureg_MUL(shader, temp[2], x,
            ureg_imm1f(shader, 3.141592));
   ureg_DIV(shader, temp[3], ureg_src(temp[2]), a);

   ureg_SIN(shader, temp[4], ureg_src(temp[2]));
   ureg_SIN(shader, temp[5], ureg_src(temp[3]));

   ureg_MUL(shader, temp[6], ureg_src(temp[4]),
            ureg_src(temp[5]));
   ureg_MUL(shader, temp[7], ureg_imm1f(shader,
            0.101321), a);
   ureg_MUL(shader, temp[7], ureg_src(temp[7]),
            ureg_src(temp[6]));
   ureg_DIV(shader, temp[7], ureg_src(temp[7]),
            ureg_src(temp[0]));
   ureg_DIV(shader, o_fragment,
           ureg_src(temp[7]), ureg_src(temp[0]));

   ureg_LRP(shader, o_fragment, ureg_src(temp[1]),
            ureg_imm1f(shader, 1.0f), ureg_src(o_fragment));

   for(i = 0; i < 8; ++i)
       ureg_release_temporary(shader, temp[i]);
}

static void *
create_frag_shader(struct vl_lanczos_filter *filter, unsigned num_offsets,
                   struct vertex2f *offsets, unsigned a,
                   unsigned video_width, unsigned video_height)
{
   struct pipe_screen *screen = filter->pipe->screen;
   struct ureg_program *shader;
   struct ureg_src i_vtex, vtex;
   struct ureg_src sampler;
   struct ureg_src half_pixel;
   struct ureg_dst o_fragment;
   struct ureg_dst *t_array = MALLOC(sizeof(struct ureg_dst) * (num_offsets + 2));
   struct ureg_dst x, t_sum;
   unsigned i;
   bool first;

   if (screen->get_shader_param(
      screen, PIPE_SHADER_FRAGMENT, PIPE_SHADER_CAP_MAX_TEMPS) < num_offsets + 2) {
      return NULL;
   }

   shader = ureg_create(PIPE_SHADER_FRAGMENT);
   if (!shader) {
      return NULL;
   }

   i_vtex = ureg_DECL_fs_input(shader, TGSI_SEMANTIC_GENERIC, VS_O_VTEX, TGSI_INTERPOLATE_LINEAR);
   sampler = ureg_DECL_sampler(shader, 0);

   for (i = 0; i < num_offsets + 2; ++i)
      t_array[i] = ureg_DECL_temporary(shader);
   x = ureg_DECL_temporary(shader);

   half_pixel = ureg_DECL_constant(shader, 0);
   o_fragment = ureg_DECL_output(shader, TGSI_SEMANTIC_COLOR, 0);

   /*
    * temp = (i_vtex * i_size)
    * x = frac(temp)
    * vtex = floor(i_vtex)/i_size - half_pixel
    */
   ureg_MUL(shader, ureg_writemask(t_array[0], TGSI_WRITEMASK_XY),
            i_vtex, ureg_imm2f(shader, video_width, video_height));
   ureg_FRC(shader, ureg_writemask(x, TGSI_WRITEMASK_XY),
            ureg_src(t_array[0]));

   ureg_FLR(shader, ureg_writemask(t_array[1], TGSI_WRITEMASK_XY),
            ureg_src(t_array[0]));
   ureg_DIV(shader, ureg_writemask(t_array[1], TGSI_WRITEMASK_XY),
            ureg_src(t_array[1]), ureg_imm2f(shader, video_width, video_height));
   ureg_SUB(shader, ureg_writemask(t_array[1], TGSI_WRITEMASK_XY),
            ureg_src(t_array[1]), half_pixel);
   /*
    * t_array[2..*] = vtex + offset[0..*]
    * t_array[2..*] = tex(t_array[0..*], sampler)
    * o_fragment = sum(t_array[i] * lanczos(x - offsets[i].x) * lanczos(y - offsets[i].y))
    */
   vtex = ureg_src(t_array[1]);
   for (i = 0; i < num_offsets; ++i) {
        ureg_ADD(shader, ureg_writemask(t_array[i + 2], TGSI_WRITEMASK_XY),
                  vtex, ureg_imm2f(shader, offsets[i].x, offsets[i].y));
        ureg_MOV(shader, ureg_writemask(t_array[i + 2], TGSI_WRITEMASK_ZW),
                  ureg_imm1f(shader, 0.0f));
   }

   for (i = 0; i < num_offsets; ++i) {
      ureg_TEX(shader, t_array[i + 2], TGSI_TEXTURE_2D, ureg_src(t_array[i + 2]), sampler);
   }

   for(i = 0, first = true; i < num_offsets; ++i) {
      if (first) {
         t_sum = t_array[i];
         ureg_SUB(shader, ureg_writemask(t_array[i], TGSI_WRITEMASK_XY),
                  ureg_src(x), ureg_imm2f(shader, offsets[i].x * video_width,
                  offsets[i].y * video_height));
         create_frag_shader_lanczos(shader, ureg_imm1f(shader, (float)(a)),
                 ureg_scalar(ureg_src(t_array[i]), TGSI_SWIZZLE_X), t_array[i + 1]);
         create_frag_shader_lanczos(shader, ureg_imm1f(shader, (float)(a)),
                 ureg_scalar(ureg_src(t_array[i]), TGSI_SWIZZLE_Y), t_array[i]);
         ureg_MUL(shader, t_array[i + 1], ureg_src(t_array[i + 1]),
                  ureg_src(t_array[i]));
         ureg_MUL(shader, t_sum, ureg_src(t_array[i + 2]),
                  ureg_src(t_array[i + 1]));
         first = false;
      } else {
         ureg_SUB(shader, ureg_writemask(t_array[i], TGSI_WRITEMASK_XY),
                  ureg_src(x), ureg_imm2f(shader, offsets[i].x * video_width,
                  offsets[i].y * video_height));
         create_frag_shader_lanczos(shader, ureg_imm1f(shader, (float)(a)),
                 ureg_scalar(ureg_src(t_array[i]), TGSI_SWIZZLE_X), t_array[i + 1]);
         create_frag_shader_lanczos(shader, ureg_imm1f(shader, (float)(a)),
                 ureg_scalar(ureg_src(t_array[i]), TGSI_SWIZZLE_Y), t_array[i]);
         ureg_MUL(shader, t_array[i + 1], ureg_src(t_array[i + 1]),
                  ureg_src(t_array[i]));
         ureg_MAD(shader, t_sum, ureg_src(t_array[i + 2]),
                  ureg_src(t_array[i + 1]), ureg_src(t_sum));
      }
   }

   if (first)
      ureg_MOV(shader, o_fragment, ureg_imm1f(shader, 0.0f));
   else
      ureg_MOV(shader, o_fragment, ureg_src(t_sum));

   ureg_release_temporary(shader, x);
   ureg_END(shader);

   FREE(t_array);
   return ureg_create_shader_and_destroy(shader, filter->pipe);
}

bool
vl_lanczos_filter_init(struct vl_lanczos_filter *filter, struct pipe_context *pipe,
                       unsigned size, unsigned width, unsigned height)
{
   struct pipe_rasterizer_state rs_state;
   struct pipe_blend_state blend;
   struct vertex2f *offsets, v, sizes;
   struct pipe_sampler_state sampler;
   struct pipe_vertex_element ve;
   unsigned i, num_offsets = (2 * size) * (2 * size);

   assert(filter && pipe);
   assert(width && height);
   assert(size);

   memset(filter, 0, sizeof(*filter));
   filter->pipe = pipe;

   memset(&rs_state, 0, sizeof(rs_state));
   rs_state.half_pixel_center = true;
   rs_state.bottom_edge_rule = true;
   rs_state.depth_clip = 1;
   filter->rs_state = pipe->create_rasterizer_state(pipe, &rs_state);
   if (!filter->rs_state)
      goto error_rs_state;

   memset(&blend, 0, sizeof blend);
   blend.rt[0].rgb_func = PIPE_BLEND_ADD;
   blend.rt[0].rgb_src_factor = PIPE_BLENDFACTOR_ONE;
   blend.rt[0].rgb_dst_factor = PIPE_BLENDFACTOR_ONE;
   blend.rt[0].alpha_func = PIPE_BLEND_ADD;
   blend.rt[0].alpha_src_factor = PIPE_BLENDFACTOR_ONE;
   blend.rt[0].alpha_dst_factor = PIPE_BLENDFACTOR_ONE;
   blend.logicop_func = PIPE_LOGICOP_CLEAR;
   blend.rt[0].colormask = PIPE_MASK_RGBA;
   filter->blend = pipe->create_blend_state(pipe, &blend);
   if (!filter->blend)
      goto error_blend;

   memset(&sampler, 0, sizeof(sampler));
   sampler.wrap_s = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   sampler.wrap_t = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   sampler.wrap_r = PIPE_TEX_WRAP_CLAMP_TO_EDGE;
   sampler.min_img_filter = PIPE_TEX_FILTER_NEAREST;
   sampler.min_mip_filter = PIPE_TEX_MIPFILTER_NONE;
   sampler.mag_img_filter = PIPE_TEX_FILTER_NEAREST;
   sampler.compare_mode = PIPE_TEX_COMPARE_NONE;
   sampler.compare_func = PIPE_FUNC_ALWAYS;
   sampler.normalized_coords = 1;
   filter->sampler = pipe->create_sampler_state(pipe, &sampler);
   if (!filter->sampler)
      goto error_sampler;

   filter->quad = vl_vb_upload_quads(pipe);
   if(!filter->quad.buffer)
      goto error_quad;

   memset(&ve, 0, sizeof(ve));
   ve.src_offset = 0;
   ve.instance_divisor = 0;
   ve.vertex_buffer_index = 0;
   ve.src_format = PIPE_FORMAT_R32G32_FLOAT;
   filter->ves = pipe->create_vertex_elements_state(pipe, 1, &ve);
   if (!filter->ves)
      goto error_ves;

   offsets = MALLOC(sizeof(struct vertex2f) * num_offsets);
   if (!offsets)
      goto error_offsets;

   sizes.x = (float)(size);
   sizes.y = (float)(size);

   for (v.y = -sizes.y + 1.0f, i = 0; v.y <= sizes.y; v.y += 1.0f)
      for (v.x = -sizes.x + 1.0f; v.x <= sizes.x; v.x += 1.0f)
         offsets[i++] = v;

   for (i = 0; i < num_offsets; ++i) {
      offsets[i].x /= width;
      offsets[i].y /= height;
   }

   filter->vs = create_vert_shader(filter);
   if (!filter->vs)
      goto error_vs;

   filter->fs = create_frag_shader(filter, num_offsets, offsets, size, width, height);
   if (!filter->fs)
      goto error_fs;

   FREE(offsets);
   return true;

error_fs:
   pipe->delete_vs_state(pipe, filter->vs);

error_vs:
   FREE(offsets);

error_offsets:
   pipe->delete_vertex_elements_state(pipe, filter->ves);

error_ves:
   pipe_resource_reference(&filter->quad.buffer, NULL);

error_quad:
   pipe->delete_sampler_state(pipe, filter->sampler);

error_sampler:
   pipe->delete_blend_state(pipe, filter->blend);

error_blend:
   pipe->delete_rasterizer_state(pipe, filter->rs_state);

error_rs_state:
   return false;
}

void
vl_lanczos_filter_cleanup(struct vl_lanczos_filter *filter)
{
   assert(filter);

   filter->pipe->delete_sampler_state(filter->pipe, filter->sampler);
   filter->pipe->delete_blend_state(filter->pipe, filter->blend);
   filter->pipe->delete_rasterizer_state(filter->pipe, filter->rs_state);
   filter->pipe->delete_vertex_elements_state(filter->pipe, filter->ves);
   pipe_resource_reference(&filter->quad.buffer, NULL);

   filter->pipe->delete_vs_state(filter->pipe, filter->vs);
   filter->pipe->delete_fs_state(filter->pipe, filter->fs);
}

void
vl_lanczos_filter_render(struct vl_lanczos_filter *filter,
                        struct pipe_sampler_view *src,
                        struct pipe_surface *dst,
                        struct u_rect *dst_area,
                        struct u_rect *dst_clip)
{
   struct pipe_viewport_state viewport;
   struct pipe_framebuffer_state fb_state;
   struct pipe_scissor_state scissor;
   union pipe_color_union clear_color;
   struct pipe_transfer *buf_transfer;
   struct pipe_resource *surface_size;
   assert(filter && src && dst);

   if (dst_clip) {
      scissor.minx = dst_clip->x0;
      scissor.miny = dst_clip->y0;
      scissor.maxx = dst_clip->x1;
      scissor.maxy = dst_clip->y1;
   } else {
      scissor.minx = 0;
      scissor.miny = 0;
      scissor.maxx = dst->width;
      scissor.maxy = dst->height;
   }

   clear_color.f[0] = clear_color.f[1] = 0.0f;
   clear_color.f[2] = clear_color.f[3] = 0.0f;
   surface_size = pipe_buffer_create
   (
      filter->pipe->screen,
      PIPE_BIND_CONSTANT_BUFFER,
      PIPE_USAGE_DEFAULT,
      2*sizeof(float)
   );


   memset(&viewport, 0, sizeof(viewport));
   if(dst_area){
      viewport.scale[0] = dst_area->x1 - dst_area->x0;
      viewport.scale[1] = dst_area->y1 - dst_area->y0;
      viewport.translate[0] = dst_area->x0;
      viewport.translate[1] = dst_area->y0;
   } else {
      viewport.scale[0] = dst->width;
      viewport.scale[1] = dst->height;
   }
   viewport.scale[2] = 1;

   float *ptr = pipe_buffer_map(filter->pipe, surface_size,
                               PIPE_TRANSFER_WRITE | PIPE_TRANSFER_DISCARD_RANGE,
                               &buf_transfer);

   ptr[0] = 0.5f/viewport.scale[0];
   ptr[1] = 0.5f/viewport.scale[1];

   pipe_buffer_unmap(filter->pipe, buf_transfer);

   memset(&fb_state, 0, sizeof(fb_state));
   fb_state.width = dst->width;
   fb_state.height = dst->height;
   fb_state.nr_cbufs = 1;
   fb_state.cbufs[0] = dst;

   filter->pipe->set_scissor_states(filter->pipe, 0, 1, &scissor);
   filter->pipe->clear_render_target(filter->pipe, dst, &clear_color,
                                     0, 0, dst->width, dst->height, false);
   pipe_set_constant_buffer(filter->pipe, PIPE_SHADER_FRAGMENT, 0, surface_size);
   filter->pipe->bind_rasterizer_state(filter->pipe, filter->rs_state);
   filter->pipe->bind_blend_state(filter->pipe, filter->blend);
   filter->pipe->bind_sampler_states(filter->pipe, PIPE_SHADER_FRAGMENT,
                                     0, 1, &filter->sampler);
   filter->pipe->set_sampler_views(filter->pipe, PIPE_SHADER_FRAGMENT,
                                   0, 1, &src);
   filter->pipe->bind_vs_state(filter->pipe, filter->vs);
   filter->pipe->bind_fs_state(filter->pipe, filter->fs);
   filter->pipe->set_framebuffer_state(filter->pipe, &fb_state);
   filter->pipe->set_viewport_states(filter->pipe, 0, 1, &viewport);
   filter->pipe->set_vertex_buffers(filter->pipe, 0, 1, &filter->quad);
   filter->pipe->bind_vertex_elements_state(filter->pipe, filter->ves);

   util_draw_arrays(filter->pipe, PIPE_PRIM_QUADS, 0, 4);
}
