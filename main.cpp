#include <igl/Camera.h>
#include <igl/REDRUM.h>
#include <igl/REDRUM.h>
#include <igl/adjacency_list.h>
#include <igl/adjacency_matrix.h>
#include <igl/boundary_conditions.h>
#include <igl/bone_parents.h>
#include <igl/centroid.h>
#include <igl/colon.h>
#include <igl/opengl2/draw_floor.h>
#include <igl/opengl2/draw_mesh.h>
#include <igl/opengl2/draw_skeleton_3d.h>
#include <igl/opengl2/draw_skeleton_vector_graphics.h>
#include <igl/file_exists.h>
#include <igl/forward_kinematics.h>
#include <igl/get_seconds.h>
#include <igl/lbs_matrix.h>
#include <igl/list_to_matrix.h>
#include <igl/material_colors.h>
#include <igl/matlab_format.h>
#include <igl/normalize_row_sums.h>
#include <igl/pathinfo.h>
#include <igl/per_face_normals.h>
#include <igl/opengl2/project.h>
#include <igl/quat_to_mat.h>
#include <igl/readBF.h>
#include <igl/readTGF.h>
#include <igl/read_triangle_mesh.h>
#include <igl/remove_unreferenced.h>
#include <igl/opengl/report_gl_error.h>
#include <igl/opengl2/right_axis.h>
#include <igl/slice.h>
#include <igl/snap_to_canonical_view_quat.h>
#include <igl/snap_to_fixed_up.h>
#include <igl/opengl2/sort_triangles.h>
#include <igl/trackball.h>
#include <igl/two_axis_valuator_fixed_up.h>
#include <igl/unique.h>
#include <igl/unique_rows.h>
#include <igl/opengl2/unproject.h>
#include <igl/opengl2/model_proj_viewport.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/writeTGF.h>
#include <igl/writeBF.h>
#include <igl/unproject_in_mesh.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

#ifndef GLUT_WHEEL_UP
#define GLUT_WHEEL_UP    3
#endif
#ifndef GLUT_WHEEL_DOWN
#define GLUT_WHEEL_DOWN  4
#endif
#ifndef GLUT_WHEEL_RIGHT
#define GLUT_WHEEL_RIGHT 5
#endif
#ifndef GLUT_WHEEL_LEFT
#define GLUT_WHEEL_LEFT  6
#endif
#ifndef GLUT_ACTIVE_COMMAND
#define GLUT_ACTIVE_COMMAND 8
#endif

#include <string>
#include <vector>
#include <queue>
#include <stack>
#include <iostream>

enum SkelStyleType
{
  SKEL_STYLE_TYPE_3D = 0,
  SKEL_STYLE_TYPE_VECTOR_GRAPHICS = 1,
  NUM_SKEL_STYLE_TYPE = 2
}skel_style;

Eigen::MatrixXd V,N,sorted_N;
Eigen::Vector3d Vmid,Vcen;
double bbd = 1.0;
Eigen::MatrixXi F,sorted_F;
Eigen::VectorXi P;
igl::Camera camera;
struct State
{
  Eigen::MatrixXd C;
  Eigen::MatrixXi BE;
  Eigen::VectorXi sel;
} s;

bool wireframe = false;
bool skeleton_on_top = false;
double alpha = 0.5;

// See README for descriptions
enum RotationType
{
  ROTATION_TYPE_IGL_TRACKBALL = 0,
  ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP = 1,
  NUM_ROTATION_TYPES = 2,
} rotation_type = ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP;

std::stack<State> undo_stack;
std::stack<State> redo_stack;

bool is_rotating = false;
bool is_dragging = false;
bool new_leaf_on_drag = false;
bool new_root_on_drag = false;
int down_x,down_y;
Eigen::MatrixXd down_C;
igl::Camera down_camera;
std::string output_filename;

bool is_animating = false;
double animation_start_time = 0;
double ANIMATION_DURATION = 0.5;
Eigen::Quaterniond animation_from_quat;
Eigen::Quaterniond animation_to_quat;

int width,height;
Eigen::Vector4f light_pos(-0.1,-0.1,0.9,0);

void push_undo()
{
  undo_stack.push(s);
  // Clear
  redo_stack = std::stack<State>();
}

// No-op setter, does nothing
void set_rotation_type(const RotationType & value)
{
  using namespace Eigen;
  using namespace std;
  using namespace igl;
  const RotationType old_rotation_type = rotation_type;
  rotation_type = value;
  if(rotation_type == ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP &&
    old_rotation_type != ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP)
  {
    push_undo();
    animation_from_quat = camera.m_rotation_conj;
    snap_to_fixed_up(animation_from_quat,animation_to_quat);
    // start animation
    animation_start_time = get_seconds();
    is_animating = true;
  }
}

void reshape(int width, int height)
{
  ::width = width;
  ::height = height;
  glViewport(0,0,width,height);
  camera.m_aspect = (double)width/(double)height;
}

void push_scene()
{
  using namespace igl;
  using namespace std;
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();
  gluPerspective(camera.m_angle,camera.m_aspect,camera.m_near,camera.m_far);
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();
  gluLookAt(
    camera.eye()(0), camera.eye()(1), camera.eye()(2),
    camera.at()(0), camera.at()(1), camera.at()(2),
    camera.up()(0), camera.up()(1), camera.up()(2));
}

void push_object()
{
  using namespace igl;
  glPushMatrix();
  glScaled(2./bbd,2./bbd,2./bbd);
  glTranslated(-Vmid(0),-Vmid(1),-Vmid(2));
}

void pop_object()
{
  glPopMatrix();
}

void pop_scene()
{
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();
}

// Set up double-sided lights
void lights()
{
  using namespace std;
  using namespace Eigen;
  glEnable(GL_LIGHTING);
  glLightModelf(GL_LIGHT_MODEL_TWO_SIDE,GL_TRUE);
  glEnable(GL_LIGHT0);
  glEnable(GL_LIGHT1);
  float WHITE[4] =  {0.8,0.8,0.8,1.};
  float GREY[4] =  {0.4,0.4,0.4,1.};
  float BLACK[4] =  {0.,0.,0.,1.};
  Vector4f pos = light_pos;
  glLightfv(GL_LIGHT0,GL_AMBIENT,GREY);
  glLightfv(GL_LIGHT0,GL_DIFFUSE,WHITE);
  glLightfv(GL_LIGHT0,GL_SPECULAR,BLACK);
  glLightfv(GL_LIGHT0,GL_POSITION,pos.data());
  pos(0) *= -1;
  pos(1) *= -1;
  pos(2) *= -1;
  glLightfv(GL_LIGHT1,GL_AMBIENT,GREY);
  glLightfv(GL_LIGHT1,GL_DIFFUSE,WHITE);
  glLightfv(GL_LIGHT1,GL_SPECULAR,BLACK);
  glLightfv(GL_LIGHT1,GL_POSITION,pos.data());
}

void sort()
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;
  push_scene();
  push_object();
  VectorXi I;
  igl::opengl2::sort_triangles(V,F,sorted_F,I);
  slice(N,I,1,sorted_N);
  pop_object();
  pop_scene();
}

void display()
{
  using namespace igl;
  using namespace std;
  using namespace Eigen;
  const float back[4] = {0.75, 0.75, 0.75,0};
  glClearColor(back[0],back[1],back[2],0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  static bool first = true;
  if(first)
  {
    sort();
    first = false;
  }

  if(is_animating)
  {
    double t = (get_seconds() - animation_start_time)/ANIMATION_DURATION;
    if(t > 1)
    {
      t = 1;
      is_animating = false;
    }
    Quaterniond q = animation_from_quat.slerp(t,animation_to_quat).normalized();
    camera.orbit(q.conjugate());
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glEnable(GL_NORMALIZE);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  lights();
  push_scene();

  // Draw a nice floor
  glEnable(GL_DEPTH_TEST);
  glPushMatrix();
  const double floor_offset =
    -2./bbd*(V.col(1).maxCoeff()-Vmid(1));
  glTranslated(0,floor_offset,0);
  const float GREY[4] = {0.5,0.5,0.6,1.0};
  const float DARK_GREY[4] = {0.2,0.2,0.3,1.0};
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);
  igl::opengl2::draw_floor(GREY,DARK_GREY);
  glDisable(GL_CULL_FACE);
  glPopMatrix();

  push_object();

  const auto & draw_skeleton = []()
  {
    switch(skel_style)
    {
      default:
      case SKEL_STYLE_TYPE_3D:
      {
        MatrixXf colors = MAYA_VIOLET.transpose().replicate(s.BE.rows(),1);
        for(int si=0;si<s.sel.size();si++)
        {
          for(int b=0;b<s.BE.rows();b++)
          {
            if(s.BE(b,0) == s.sel(si) || s.BE(b,1) == s.sel(si))
            {
              colors.row(b) = MAYA_SEA_GREEN;
            }
          }
        }
        igl::opengl2::draw_skeleton_3d(s.C,s.BE,MatrixXd(),colors,bbd*0.5);
        break;
      }
      case SKEL_STYLE_TYPE_VECTOR_GRAPHICS:
        igl::opengl2::draw_skeleton_vector_graphics(s.C,s.BE);
        break;
    }
  };
  
  if(!skeleton_on_top)
  {
    draw_skeleton();
  }

  // Set material properties
  glDisable(GL_COLOR_MATERIAL);
  glMaterialfv(GL_FRONT, GL_AMBIENT,
    Vector4f(GOLD_AMBIENT[0],GOLD_AMBIENT[1],GOLD_AMBIENT[2],alpha).data());
  glMaterialfv(GL_FRONT, GL_DIFFUSE,
    Vector4f(GOLD_DIFFUSE[0],GOLD_DIFFUSE[1],GOLD_DIFFUSE[2],alpha).data());
  glMaterialfv(GL_FRONT, GL_SPECULAR,
    Vector4f(GOLD_SPECULAR[0],GOLD_SPECULAR[1],GOLD_SPECULAR[2],alpha).data());
  glMaterialf (GL_FRONT, GL_SHININESS, 128);
  glMaterialfv(GL_BACK, GL_AMBIENT,
    Vector4f(SILVER_AMBIENT[0],SILVER_AMBIENT[1],SILVER_AMBIENT[2],alpha).data());
  glMaterialfv(GL_BACK, GL_DIFFUSE,
    Vector4f(FAST_GREEN_DIFFUSE[0],FAST_GREEN_DIFFUSE[1],FAST_GREEN_DIFFUSE[2],alpha).data());
  glMaterialfv(GL_BACK, GL_SPECULAR,
    Vector4f(SILVER_SPECULAR[0],SILVER_SPECULAR[1],SILVER_SPECULAR[2],alpha).data());
  glMaterialf (GL_BACK, GL_SHININESS, 128);

  if(wireframe)
  {
    glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
  }
  glLineWidth(1.0);
  igl::opengl2::draw_mesh(V,sorted_F,sorted_N);
  glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);

  if(skeleton_on_top)
  {
    glDisable(GL_DEPTH_TEST);
    draw_skeleton();
  }

  pop_object();
  pop_scene();

  igl::opengl::report_gl_error();

  glutSwapBuffers();
  if(is_animating)
  {
    glutPostRedisplay();
  }
}

void mouse_wheel(int wheel, int direction, int /*x*/, int /*y*/)
{
  using namespace std;
  using namespace igl;
  using namespace Eigen;
  GLint viewport[4];
  glGetIntegerv(GL_VIEWPORT,viewport);
  push_undo();

  if(wheel==0)
  {
    // factor of zoom change
    double s = (1.-0.01*direction);
    //// FOV zoom: just widen angle. This is hardly ever appropriate.
    //camera.m_angle *= s;
    //camera.m_angle = min(max(camera.m_angle,1),89);
    camera.push_away(s);
  }else
  {
    // Dolly zoom:
    camera.dolly_zoom((double)direction*1.0);
  }
  glutPostRedisplay();
}

Eigen::VectorXi selection(const std::vector<bool> & mask)
{
  const int count = std::count(mask.begin(),mask.end(),true);
  Eigen::VectorXi sel(count);
  int s = 0;
  for(int c = 0;c<(int)mask.size();c++)
  {
    if(mask[c])
    {
      sel(s) = c;
      s++;
    }
  }
  return sel;
}

std::vector<bool> selection_mask(const Eigen::VectorXi & sel, const int n)
{
  std::vector<bool> mask(n,false);
  for(int si = 0;si<sel.size();si++)
  {
    const int i = sel(si);
    mask[i] = true;
  }
  return mask;
}

bool ss_select(
  const double mouse_x, 
  const double mouse_y,
  const Eigen::MatrixXd & C,
  const bool accum,
  Eigen::VectorXi & sel)
{
  using namespace igl;
  using namespace Eigen;
  using namespace std;
  //// zap old list
  //if(!accum)
  //{
  //  sel.resize(0,1);
  //}

  vector<bool> old_mask = selection_mask(s.sel,s.C.rows());
  vector<bool> mask(old_mask.size(),false);

  double min_dist = 1e25;
  bool sel_changed = false;
  bool found = false;
  for(int c = 0;c<C.rows();c++)
  {
    const RowVector3d & Cc = C.row(c);
    const auto Pc = igl::opengl2::project(Cc);
    const double SELECTION_DIST = 18;//pixels
    const double dist = (Pc.head(2)-RowVector2d(mouse_x,height-mouse_y)).norm();
    if(dist < SELECTION_DIST && (accum || dist < min_dist))
    {
      mask[c] = true;
      min_dist = dist;
      found = true;
      sel_changed |= mask[c] != old_mask[c];
    }
  }
  for(int c = 0;c<C.rows();c++)
  {
    if(accum)
    {
      mask[c] = mask[c] ^ old_mask[c];
    }else
    {
      if(!sel_changed)
      {
        mask[c] = mask[c] || old_mask[c];
      }
    }
  }
  sel = selection(mask);
  return found;
}

void mouse(int glutButton, int glutState, int mouse_x, int mouse_y)
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;
  const int mod = (glutButton <=2 ? glutGetModifiers() : 0);
  const bool option_down = mod & GLUT_ACTIVE_ALT;
  const bool shift_down = mod & GLUT_ACTIVE_SHIFT;
  const bool command_down = GLUT_ACTIVE_COMMAND & mod;
  switch(glutButton)
  {
    case GLUT_RIGHT_BUTTON:
    case GLUT_LEFT_BUTTON:
    {
      switch(glutState)
      {
        case 1:
          // up
          glutSetCursor(GLUT_CURSOR_INHERIT);
          if(is_rotating)
          {
            sort();
          }
          is_rotating = false;
          is_dragging = false;
          break;
        case 0:
          new_leaf_on_drag = false;
          new_root_on_drag = false;
          {
            down_x = mouse_x;
            down_y = mouse_y;
            if(option_down)
            {
              glutSetCursor(GLUT_CURSOR_CYCLE);
              // collect information for trackball
              is_rotating = true;
              down_camera = camera;
            }else
            {
              push_undo();
              push_scene();
              push_object();
              // Zap selection
              if(shift_down)
              {
                s.sel.resize(0,1);
              }
              if(ss_select(mouse_x,mouse_y,s.C,
                command_down && !shift_down,
                s.sel))
              {
                if(shift_down)
                {
                  new_leaf_on_drag = true;
                }
              }else
              {
                new_root_on_drag = true;
              }
              is_dragging = !command_down;
              down_C = s.C;
              pop_object();
              pop_scene();
            }
          }
        break;
      }
      break;
    }
    // Scroll down
    case 3:
    {
      mouse_wheel(0,-1,mouse_x,mouse_y);
      break;
    }
    // Scroll up
    case 4:
    {
      mouse_wheel(0,1,mouse_x,mouse_y);
      break;
    }
    // Scroll left
    case 5:
    {
      mouse_wheel(1,-1,mouse_x,mouse_y);
      break;
    }
    // Scroll right
    case 6:
    {
      mouse_wheel(1,1,mouse_x,mouse_y);
      break;
    }
  }
  glutPostRedisplay();
}

void mouse_drag(int mouse_x, int mouse_y)
{
  using namespace igl;
  using namespace std;
  using namespace Eigen;

  if(is_rotating)
  {
    glutSetCursor(GLUT_CURSOR_CYCLE);
    Quaterniond q;
    switch(rotation_type)
    {
      case ROTATION_TYPE_IGL_TRACKBALL:
      {
        // Rotate according to trackball
        igl::trackball<double>(
          width,
          height,
          2.0,
          down_camera.m_rotation_conj.coeffs().data(),
          down_x,
          down_y,
          mouse_x,
          mouse_y,
          q.coeffs().data());
          break;
      }
      case ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP:
      {
        // Rotate according to two axis valuator with fixed up vector
        two_axis_valuator_fixed_up(
          width, height,
          2.0,
          down_camera.m_rotation_conj,
          down_x, down_y, mouse_x, mouse_y,
          q);
        break;
      }
      default:
        break;
    }
    camera.orbit(q.conjugate());
  }

  if(is_dragging)
  {
    push_scene();
    push_object();
    if(new_leaf_on_drag)
    {
      assert(s.C.size() >= 1);
      // one new node
      s.C.conservativeResize(s.C.rows()+1,3);
      const int nc = s.C.rows();
      assert(s.sel.size() >= 1);
      s.C.row(nc-1) = s.C.row(s.sel(0));
      // one new bone
      s.BE.conservativeResize(s.BE.rows()+1,2);
      s.BE.row(s.BE.rows()-1) = RowVector2i(s.sel(0),nc-1);
      // select just last node
      s.sel.resize(1,1);
      s.sel(0) = nc-1;
      // reset down_C
      down_C = s.C;
      new_leaf_on_drag = false;
    }
      Eigen::Matrix4f model,proj;
      Eigen::Vector4f viewport;
      igl::opengl2::model_proj_viewport(model,proj,viewport);
      Eigen::Vector2f pos(mouse_x,height-mouse_y);
    if(new_root_on_drag)
    {
      // two new nodes
      s.C.conservativeResize(s.C.rows()+2,3);
      const int nc = s.C.rows();
      Vector3d obj;

      int nhits = igl::unproject_in_mesh(
          pos,model,proj,viewport,V,F,obj);

      if(nhits == 0)
      {
        Vector3d pV_mid = igl::opengl2::project(Vcen);
        obj = igl::opengl2::unproject(Vector3d(mouse_x,height-mouse_y,pV_mid(2)));
      }
      s.C.row(nc-2) = obj;
      s.C.row(nc-1) = obj;
      // select last node
      s.sel.resize(1,1);
      s.sel(0) = nc-1;
      // one new bone
      s.BE.conservativeResize(s.BE.rows()+1,2);
      s.BE.row(s.BE.rows()-1) = RowVector2i(nc-2,nc-1);
      // reset down_C
      down_C = s.C;
      new_root_on_drag = false;
    }
    double z = 0;
    Vector3d obj,win;
    int nhits = igl::unproject_in_mesh(pos,model,proj,viewport,V,F,obj);
    igl::opengl2::project(obj,win);
    z = win(2);

    for(int si = 0;si<s.sel.size();si++)
    {
      const int c = s.sel(si);
      Vector3d pc = igl::opengl2::project((RowVector3d) down_C.row(c));
      pc(0) += mouse_x-down_x;
      pc(1) += (height-mouse_y)-(height-down_y);
      if(nhits > 0)
      {
        pc(2) = z;
      }
      s.C.row(c) = igl::opengl2::unproject(pc);
    }
    pop_object();
    pop_scene();
  }

  glutPostRedisplay();
}

void init_relative()
{
  using namespace Eigen;
  using namespace igl;
  using namespace std;
  per_face_normals(V,F,N);
  const auto Vmax = V.colwise().maxCoeff();
  const auto Vmin = V.colwise().minCoeff();
  Vmid = 0.5*(Vmax + Vmin);
  centroid(V,F,Vcen);
  bbd = (Vmax-Vmin).norm();
  camera.push_away(2);
}

void undo()
{
  using namespace std;
  if(!undo_stack.empty())
  {
    redo_stack.push(s);
    s = undo_stack.top();
    undo_stack.pop();
  }
}

void redo()
{
  using namespace std;
  if(!redo_stack.empty())
  {
    undo_stack.push(s);
    s = redo_stack.top();
    redo_stack.pop();
  }
}

void symmetrize()
{
  using namespace std;
  using namespace igl;
  using namespace Eigen;
  if(s.sel.size() == 0)
  {
    cout<<YELLOWGIN("Make a selection first.")<<endl;
    return;
  }
  push_undo();
  push_scene();
  push_object();
  Vector3d right;
  igl::opengl2::right_axis(right.data(),right.data()+1,right.data()+2);
  right.normalize();
  MatrixXd RC(s.C.rows(),s.C.cols());
  MatrixXd old_C = s.C;
  for(int c = 0;c<s.C.rows();c++)
  {
    const Vector3d Cc = s.C.row(c);
    const auto A = Cc-Vcen;
    const auto A1 = A.dot(right) * right;
    const auto A2 = A-A1;
    RC.row(c) = Vcen + A2 - A1;
  }
  vector<bool> mask = selection_mask(s.sel,s.C.rows());
  // stupid O(n²) matching
  for(int c = 0;c<s.C.rows();c++)
  {
    // not selected
    if(!mask[c])
    {
      continue;
    }
    const Vector3d Cc = s.C.row(c);
    int min_r = -1;
    double min_dist = 1e25;
    double max_dist = 0.1*bbd;
    for(int r= 0;r<RC.rows();r++)
    {
      const Vector3d RCr = RC.row(r);
      const double dist = (Cc-RCr).norm();
      if(
          dist<min_dist &&  // closest
          dist<max_dist && // not too far away
          (c==r || (Cc-Vcen).dot(right)*(RCr-Vcen).dot(right) > 0) // on same side
        )
      {
        min_dist = dist;
        min_r = r;
      }
    }
    if(min_r>=0)
    {
      if(mask[min_r])
      {
        s.C.row(c) = 0.5*(Cc.transpose()+RC.row(min_r));
      }else
      {
        s.C.row(c) = RC.row(min_r);
      }
    }
  }
  pop_object();
  pop_scene();
}

bool save(const std::string ext)
{
  using namespace std;
  using namespace igl;
  bool success = false;
  if(ext == ".tgf")
  {
    success = writeTGF(output_filename+ext,s.C,s.BE);
  }else if(ext == ".bf")
  {
    using namespace Eigen;
    VectorXi P;
    const auto & BE = s.BE;
    const auto & C = s.C;
    bone_parents(BE,P);
    const int n = BE.rows() + (P.array() < 0).count();
    MatrixXd offsets(n,3);
    VectorXi WI(n),bfP(n);
    {
      int r = BE.rows();
      for(int b = 0;b<BE.rows();b++)
      {
        WI(b) = b;
        offsets.row(b) = C.row(BE(b,1)) - C.row(BE(b,0));
        if(P(b)<0)
        {
          // root
          bfP(b) = r;
          WI(r) = -1;
          bfP(r) = -1;
          offsets.row(r) = C.row(BE(b,0));
          r++;
        }else
        {
          // non-root
          bfP(b) = P(b);
        }
      }
    }
    {
      // remove duplicate roots
      VectorXi I,J;
      MatrixXd WIbfPoffsets;
      unique_rows(
        (MatrixXd(n,5)<<
          WI.cast<double>(),
          bfP.cast<double>(),
          offsets).finished(),
        WIbfPoffsets,
        I,
        J);
      WI = WIbfPoffsets.col(0).cast<int>();
      bfP = WIbfPoffsets.col(1).cast<int>();
      offsets = WIbfPoffsets.rightCols(3);
      for(int i = 0;i<WI.rows();i++)
      {
        if(bfP(i)>=0)
        {
          bfP(i) = J(bfP(i));
        }
      }
    }
    success = writeBF(output_filename+ext,WI,bfP,offsets);
  }
  if(success)
  {
    cout<<GREENGIN("Current skeleton written to "+output_filename+ext+".")<<endl;
    return true;
  }else
  {
    cout<<REDRUM("Writing to "+output_filename+ext+" failed.")<<endl;
    return false;
  }
}

void key(unsigned char key, int /*x*/, int /*y*/)
{
  using namespace std;
  using namespace igl;
  using namespace Eigen;
  int mod = glutGetModifiers();
  const bool command_down = GLUT_ACTIVE_COMMAND & mod;
  const bool shift_down = GLUT_ACTIVE_SHIFT & mod;
  switch(key)
  {
    // ESC
    case char(27):
    // ^C
    case char(3):
      exit(0);
    case char(127):
    {
      push_undo();
      // delete
      MatrixXi new_BE(s.BE.rows(),s.BE.cols());
      int count = 0;
      for(int b=0;b<s.BE.rows();b++)
      {
        bool selected = false;
        for(int si=0;si<s.sel.size();si++)
        {
          if(s.BE(b,0) == s.sel(si) || s.BE(b,1) == s.sel(si))
          {
            selected = true;
            break;
          }
        }
        if(!selected)
        {
          new_BE.row(count) = s.BE.row(b);
          count++;
        }
      }
      new_BE.conservativeResize(count,new_BE.cols());
      const auto old_C = s.C;
      VectorXi I;
      remove_unreferenced(old_C,new_BE,s.C,s.BE,I);
      s.sel.resize(0,1);
      break;
    }
    case 'A':
    case 'a':
    {
      push_undo();
      s.sel = colon<int>(0,s.C.rows()-1);
      break;
    }
    case 'C':
    case 'c':
    {
      push_undo();
      // snap close vertices
      SparseMatrix<double> A;
      adjacency_matrix(s.BE,A);
      VectorXi J = colon<int>(0,s.C.rows()-1);
      // stupid O(n²) version
      for(int c = 0;c<s.C.rows();c++)
      {
        for(int d = c+1;d<s.C.rows();d++)
        {
          if(
           A.coeff(c,d) == 0 &&  // no edge
           (s.C.row(c)-s.C.row(d)).norm() < 0.02*bbd //close
           )
          {
            // c < d
            J(d) = c;
          }
        }
      }
      for(int e = 0;e<s.BE.rows();e++)
      {
        s.BE(e,0) = J(s.BE(e,0));
        s.BE(e,1) = J(s.BE(e,1));
      }
      const auto old_BE = s.BE;
      const auto old_C = s.C;
      VectorXi I;
      remove_unreferenced(old_C,old_BE,s.C,s.BE,I);
      for(int i = 0;i<s.sel.size();i++)
      {
        s.sel(i) = J(s.sel(i));
      }
      VectorXi _;
      igl::unique(s.sel,s.sel,_,_);
      break;
    }
    case 'D':
    case 'd':
    {
      push_undo();
      s.sel.resize(0,1);
      break;
    }
    case 'L':
    case 'l':
    {
      wireframe = !wireframe;
      break;
    }
    case 'O':
    case 'o':
    {
      skeleton_on_top = !skeleton_on_top;
      break;
    }
    case 'P':
    case 'p':
    {
      // add bone to parents (should really only be one)
      push_undo();
      vector<int> new_sel;
      const int old_nbe = s.BE.rows();
      for(int si=0;si<s.sel.size();si++)
      {
        for(int b=0;b<old_nbe;b++)
        {
          if(s.BE(b,1) == s.sel(si))
          {
            // one new node
            s.C.conservativeResize(s.C.rows()+1,3);
            const int nc = s.C.rows();
            s.C.row(nc-1) = 0.5*(s.C.row(s.BE(b,1)) + s.C.row(s.BE(b,0)));
            // one new bone
            s.BE.conservativeResize(s.BE.rows()+1,2);
            s.BE.row(s.BE.rows()-1) = RowVector2i(nc-1,s.BE(b,1));
            s.BE(b,1) = nc-1;
            // select last node
            new_sel.push_back(nc-1);
          }
        }
      }
      list_to_matrix(new_sel,s.sel);
      break;
    }
    case 'R':
    case 'r':
    {
      // re-root try at first selected
      if(s.sel.size() > 0)
      {
        push_undo();
        // only use first
        s.sel.conservativeResize(1,1);
        // Ideally this should only effect the connected component of s.sel(0)
        const auto & C = s.C;
        auto & BE = s.BE;
        vector<bool> seen(C.rows(),false);
        // adjacency list
        vector<vector< int> > A;
        adjacency_list(BE,A,false);
        int e = 0;
        queue<int> Q;
        Q.push(s.sel(0));
        seen[s.sel(0)] = true;
        while(!Q.empty())
        {
          const int c = Q.front();
          Q.pop();
          for(const auto & d : A[c])
          {
            if(!seen[d])
            {
              BE(e,0) = c;
              BE(e,1) = d;
              e++;
              Q.push(d);
              seen[d] = true;
            }
          }
        }
        // only keep tree
        BE.conservativeResize(e,BE.cols());
      }
      break;
    }
    case 'S':
    {
      save(".bf");
      break;
    }
    case 's':
    {
      save(".tgf");
      break;
    }
    case 'U':
    case 'u':
    {
      push_scene();
      push_object();
      for(int c = 0;c<s.C.rows();c++)
      {
        Vector3d P = igl::opengl2::project((Vector3d)s.C.row(c));
        Vector3d obj;
        Eigen::Matrix4f model,proj;
        Eigen::Vector4f viewport;
        igl::opengl2::model_proj_viewport(model,proj,viewport);
        Eigen::Vector2f pos(P(0),P(1));
        int nhits = igl::unproject_in_mesh(pos,model,proj,viewport,V,F,obj);
        if(nhits > 0)
        {
          s.C.row(c) = obj;
        }
      }
      pop_object();
      pop_scene();
      break;
    }
    case 'Y':
    case 'y':
    {
      symmetrize();
      break;
    }
    case 'z':
    case 'Z':
      is_rotating = false;
      is_dragging = false;
      if(command_down)
      {
        if(shift_down)
        {
          redo();
        }else
        {
          undo();
        }
        break;
      }else
      {
        push_undo();
        Quaterniond q;
        snap_to_canonical_view_quat(camera.m_rotation_conj,1.0,q);
        camera.orbit(q.conjugate());
      }
      break;
    case '[':
    case ']':
    {
      if(rotation_type == ROTATION_TYPE_IGL_TRACKBALL)
      {
        set_rotation_type(ROTATION_TYPE_TWO_AXIS_VALUATOR_FIXED_UP);
      }else
      {
        set_rotation_type(ROTATION_TYPE_IGL_TRACKBALL);
      }
      break;
    }
    case '{':
    {
      alpha = std::max(std::min(alpha-0.1,1.),0.);
      break;
    }
    case '}':
    {
      alpha = std::max(std::min(alpha+0.1,1.),0.);
      break;
    }
    case '<':
    case '>':
    {
      if(skel_style == SKEL_STYLE_TYPE_3D)
      {
        skel_style = SKEL_STYLE_TYPE_VECTOR_GRAPHICS;
      }else
      {
        skel_style = SKEL_STYLE_TYPE_3D;
      }
      break;
    }
    default:
      {
        cout<<"Unknown key command: "<<key<<" "<<int(key)<<endl;
      }
  }

  glutPostRedisplay();
}


int main(int argc, char * argv[])
{
  using namespace std;
  using namespace Eigen;
  using namespace igl;
  string filename = "../shared/decimated-knight.obj";
  string skel_filename = "";
  output_filename = "";
  switch(argc)
  {
    case 4:
      output_filename = argv[3];
      //fall through
    case 3:
      skel_filename = argv[2];
      if(output_filename.size() == 0)
      {
        output_filename = skel_filename;
      }
      //fall through
    case 2:
      // Read and prepare mesh
      filename = argv[1];
      break;
    default:
      cerr<<"Usage:"<<endl<<"    ./example input.obj [input/output.tgf]"<<endl;
      cout<<endl<<"Opening default mesh..."<<endl;
  }
  if(output_filename.size() > 0)
  {
    // strip extension
    std::string d,b,e,f;
    pathinfo(output_filename,d,b,e,f);
    output_filename = d+"/"+f;
  }

  // print key commands
  cout<<R"(
   [Click] and [drag]  Create bone (or select node) and reposition.
⇧ +[Click] and [drag]  Select node (or create one) and _pull out_ new bone.
⌥ +[Click] and [drag]  Rotate secene.
                   ⌫   Delete selected node(s) and incident bones.
                  A,a  Select all.
                  D,d  Deselect all.
                  C,c  Snap close nodes.
                  P,p  Split \"parent\" bone(s) of selection by creating new
                       node(s).
                  R,r  Breadth first search at selection to redirect skeleton
                       into tree.
                    S  Save current skeleton to output .bf file.
                    s  Save current skeleton to output .tgf file.
                  U,u  Project then igl::opengl2::unproject inside mesh (as if
                       dragging each by ε).
                  Y,Y  Symmetrize selection over plane through object centroid
                       and right vector.
                  Z,z  Snap to canonical view.
                  ⌘ Z  Undo.
                ⇧ ⌘ Z  Redo.
                  [,]  Toggle rotation type: fixed up vs. trackball.
                  {,}  Decrease, increase alpha of model
                  <,>  Toggle skeleton visualization type: 3d maya-style vs.
                       vector graphics
               ^C,ESC  Exit (without saving).
)";

  string dir,_1,_2,name;
  read_triangle_mesh(filename,V,F,dir,_1,_2,name);

  if(output_filename.size() == 0)
  {
    output_filename = dir+"/"+name;
  }

  for(const std::string ext : {".tgf",".bf"})
  {
    if(file_exists((output_filename+ext).c_str()))
    {
      cout<<YELLOWGIN("Output set to overwrite "<<output_filename+ext)<<endl;
    }else
    {
      cout<<BLUEGIN("Output set to "<<output_filename+ext)<<endl;
    }
  }

  if(skel_filename.length() > 0)
  {
    std::string d,b,e,f;
    pathinfo(skel_filename,d,b,e,f);
    if(e == "bf")
    {
      VectorXi WI,bfP,P;
      MatrixXd offsets;
      readBF(skel_filename,WI,bfP,offsets,s.C,s.BE,P);
    }else
    {
      readTGF(skel_filename,s.C,s.BE);
    }
  }

  init_relative();

  // Init glut
  glutInit(&argc,argv);

  // Init antweakbar
  glutInitDisplayString( "rgba depth double samples>=8 ");
  glutInitWindowSize(glutGet(GLUT_SCREEN_WIDTH)/2.0,glutGet(GLUT_SCREEN_HEIGHT)/2.0);
  glutCreateWindow("skeleton-builder");
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutKeyboardFunc(key);
  glutMouseFunc(mouse);
  glutMotionFunc(mouse_drag);
  glutMainLoop();

  return 0;
}
