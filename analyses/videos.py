from bioptim import Solution
import bioviz


def generate_video(solution: Solution, save_folder: str, filename: str) -> None:
    """
    Generate a video of the solution.

    Parameters
    ----------
    solution : Solution
        The solution to be visualized.
    save_folder : str
        The folder where the video will be saved.
    filename : str
        The name of the video.
    """
    b: bioviz.Viz = solution.animate(
        show_now=False,
        show_meshes=True,
        show_global_center_of_mass=False,
        show_gravity_vector=False,
        show_floor=False,
        show_segments_center_of_mass=False,
        show_global_ref_frame=False,
        show_local_ref_frame=False,
        show_markers=True,
        show_muscles=False,
        show_wrappings=False,
        background_color=(0, 0, 0),
        mesh_opacity=0.95,
    )[0]

    b.resize(1920, 1080)

    # Position camera
    b.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
    b.set_camera_roll(90)
    b.set_camera_zoom(0.308185240948253)
    b.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)

    # Record
    b.start_recording(f"{save_folder}/{filename}")

    for f in range(sum(solution.ns) + 2):
        b.movement_slider[0].setValue(f)
        b.add_frame()
    b.stop_recording()
    b.quit()
