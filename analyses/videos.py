from bioptim import Solution
import bioviz


class Videos:
    def __init__(
        self,
        cycle_in_and_out: tuple[tuple[int, int], ...],
        camera_name_pos_roll: tuple[tuple[str, tuple[float, float, float], float], ...],
    ):
        self.cycle_in_and_out = cycle_in_and_out
        self.camera_name_pos_roll = camera_name_pos_roll

    def generate_video(self, studies, all_solutions: list[tuple[Solution, list[Solution, ...]], ...], save_folder: str):
        for study, (solution, all_iterations) in zip(studies.studies, all_solutions):
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
                background_color=(0, 1, 0),
            )[0]

            b.resize(1920, 1080)
            ns_per_cycle = study.nmpc.n_shooting_per_cycle
            for cycles in self.cycle_in_and_out:
                for name, pos, roll in self.camera_name_pos_roll:
                    # Position camera
                    b.set_camera_position(*pos)
                    b.set_camera_roll(roll)

                    # Record
                    b.start_recording(f"{save_folder}/{study.save_name}_from_{cycles[0]}_to_{cycles[1]}_{name}")
                    for f in range(cycles[0] * ns_per_cycle, cycles[1] * ns_per_cycle):
                        b.movement_slider[0].setValue(f)
                        b.add_frame()
                    b.stop_recording()
            b.quit()

    def generate_snapshot(
        self, studies, all_solutions: list[tuple[Solution, list[Solution, ...]], ...], save_folder: str
    ):
        for study, (solution, all_iterations) in zip(studies.studies, all_solutions):
            b: bioviz.Viz = solution.animate(
                show_now=False,
                show_meshes=True,
                show_global_center_of_mass=False,
                show_gravity_vector=False,
                show_floor=False,
                show_segments_center_of_mass=False,
                show_global_ref_frame=False,
                show_local_ref_frame=False,
                show_markers=False,
                show_muscles=False,
                show_wrappings=False,
                background_color=(1, 1, 1),
            )[0]

            b.resize(1920, 1080)
            ns_per_cycle = study.nmpc.n_shooting_per_cycle

            for cycles in self.cycle_in_and_out:
                for name, pos, roll in self.camera_name_pos_roll:
                    # Position camera
                    b.set_camera_position(*pos)
                    b.set_camera_roll(roll)

                    # Taking snapshotw
                    for f in range(cycles[0] * ns_per_cycle, cycles[1] * ns_per_cycle):
                        b.movement_slider[0].setValue(f)
                        b.snapshot(f"{save_folder}/{study.save_name}_{name}_{f}")

            b.quit()
