from manimlib import *
import numpy as np

# To watch one of these scenes, run the following:
# manimgl example_scenes.py OpeningManimExample
# Use -s to skip to the end and just save the final frame
# Use -w to write the animation to a file
# Use -o to write it to a file and open it once done
# Use -n <number> to skip ahead to the n'th animation of a scene.

mass_matrix = "M(q)"
mass_matrix_full = (
    "\\begin{bmatrix} M_{BB}(q) & M_{BJ}(q) "
    "\\\\"
    "M_{JB}(q) & M_{JJ}(q) \\end{bmatrix}"
)
mass_matrix_full_matrix = Matrix([["M_{BB}(q)", "M_{BJ}(q)"], ["M_{JB}(q)", "M_{JJ}(q)"]])
mass_matrix_full_matrix.get_entries()[0].set_color(BLUE)
# mass_matrix_full_matrix.get_entries()[2].set_color(RED)
mass_matrix_full_matrix.get_entries()[3].set_color(RED)

qddot = "\\ddot{q}"
qddot_full = (
    "\\begin{bmatrix} \\ddot{q}_B "
    "\\\\"
    " \\ddot{q}_J \\end{bmatrix}"
)
qddot_full_matrix = Matrix(qddot_full)
qddot_full_matrix.get_entries()[0].set_color(BLUE)
qddot_full_matrix.get_entries()[1].set_color(RED)

n = "\\begin{bmatrix} N_{B}(\\dot{q}, q) \\\\ N_{J}(\\dot{q}, q) \\end{bmatrix}"
n_matrix = Matrix(n)
n_matrix.get_entries()[0].set_color(BLUE)
n_matrix.get_entries()[1].set_color(RED)


class ForwardDynamics(Scene):
    def construct(self):
        lines = VGroup(
            # Euler-Lagrange equation
            Tex(
                "\\frac{d}{dt}\\frac{\\partial L}{\\partial \\dot{q}}",
                "-",
                "\\frac{\\partial L}{\\partial q}",
                "=",
                '\\tau',
            ),
            # Euler-lagrange in minimal coordinates form
            Tex(
                mass_matrix,
                "\\ddot{q}",
                "+",
                "N(\\dot{q},q)",
                '=',
                '\\tau'
            ),
            # forward dynamics
            Tex(
                "\\ddot{q}",
                "=",
                mass_matrix,
                "^{-1}",
                "(",
                '\\tau',
                "-",
                "N(\\dot{q},q)",
                ")",
            ),
            # separating floating base from the rest of the system
            # Tex(
            #     # M(q)
            #     mass_matrix_full,
            #     # QDDOT
            #     qddot_full,
            #     "+",
            #     # C(q, qdot)
            #     "\\begin{bmatrix} N_{B}(\\dot{q}, q) \\\\ N_{J}(\\dot{q}, q) \\end{bmatrix}",
            #     "=",
            #     "\\begin{bmatrix} 0_{6\\times1} \\\\ \\tau_J \\end{bmatrix}",
            # ),
        )
        lines.arrange(DOWN, buff=LARGE_BUFF)

        play_kw = {"run_time": 2}
        self.add(lines[0])

        self.play(
            TransformMatchingTex(
                lines[0].copy(), lines[1],
            ),
            **play_kw
        )
        self.wait()
        #
        self.play(
            TransformMatchingTex(
                lines[1].copy(), lines[2],
                key_map={
                    mass_matrix_full: mass_matrix,
                    qddot_full: qddot,
                }
            ),
            **play_kw
        )
        self.wait(duration=4)


class FreeFloatingBaseDynamics(Scene):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.minimal_euler_lagrange = Tex(
                    mass_matrix,
                    "\\ddot{q}",
                    "+",
                    "N(\\dot{q},q)",
                    '=',
                    '\\tau'
                )

        self.full_minimal_euler_lagrange = Tex(
                    # M(q)
                    mass_matrix_full,
                    # QDDOT
                    qddot_full,
                    "+",
                    # C(q, qdot)
                    "\\begin{bmatrix} N_{B}(\\dot{q}, q) \\\\ N_{J}(\\dot{q}, q) \\end{bmatrix}",
                    "=",
                    "\\begin{bmatrix} 0_{6\\times1} \\\\ \\tau_J \\end{bmatrix}",
                )
        # self.full_minimal_euler_lagrange.set_color_by_tex("M_{BB}", BLUE)
        # self.full_minimal_euler_lagrange.set_color_by_tex("M_{BJ}", GREEN)
        # self.full_minimal_euler_lagrange.set_color_by_tex("\\ddot{q}_J", BLUE)
        # self.full_minimal_euler_lagrange.set_color_by_tex("\\ddot{q}_B", RED)
        # self.full_minimal_euler_lagrange.set_color_by_tex("N_{B}", RED)

        self.free_floating_dynamics = Tex(
            "M_{BB}(q)",
            "\\ddot{q}_B",
            "+",
            "M_{BJ}(q)",
            "\\ddot{q}_J",
            "+",
            "N_{B}(\\dot{q}, q)",
            "=",
            "0_{6\\times1}",
        )
        # self.free_floating_dynamics.set_color_by_tex("M_{BB}", BLUE)
        # self.free_floating_dynamics.set_color_by_tex("M_{BJ}", GREEN)
        # self.free_floating_dynamics.set_color_by_tex("\\ddot{q}_J", BLUE)
        # self.free_floating_dynamics.set_color_by_tex("\\ddot{q}_B", RED)
        # self.free_floating_dynamics.set_color_by_tex("N_{B}", RED)

        self.free_floating_dynamics_forward = Tex(
                "\\ddot{q}_B",
                "=",
                "-",
                "M_{BB}(q)",
                "^{-1}",
                "(",
                "M_{BJ}(q)",
                "\\ddot{q}_J",
                "+",
                "N_{B}(\\dot{q}, q)",
                ")",
            )
        # self.free_floating_dynamics_forward.set_color_by_tex("M_{BB}", BLUE)
        # self.free_floating_dynamics_forward.set_color_by_tex("M_{BJ}", GREEN)
        # self.free_floating_dynamics_forward.set_color_by_tex("\\ddot{q}_J", BLUE)
        # self.free_floating_dynamics_forward.set_color_by_tex("\\ddot{q}_B", RED)
        # self.free_floating_dynamics_forward.set_color_by_tex("N_{B}", RED)

    def construct(self):
        lines = VGroup(
            # Euler-lagrange in minimal coordinates form
            self.minimal_euler_lagrange,
            # separating floating base from the rest of the system
            self.full_minimal_euler_lagrange,
            # free floating base dynamics
            self.free_floating_dynamics,
            # free floating base dynamics forward dynamics
            self.free_floating_dynamics_forward,
        )
        lines.arrange(DOWN, buff=LARGE_BUFF)

        # create a box arround the first line of the developped equation
        box = SurroundingRectangle(lines[1], buff=0.1, height=lines[1].get_height()/2)
        # blue box
        box.set_color(BLUE)

        play_kw = {"run_time": 2}
        self.add(lines[0])

        self.play(
            TransformMatchingTex(
                lines[0].copy(), lines[1],
            ),
            **play_kw
        )
        self.wait()
        # show box
        # self.play(ShowCreation(box), run_time=1)
        #
        self.play(
            TransformMatchingTex(
                lines[1].copy(), lines[2],
                key_map={
                    mass_matrix_full: mass_matrix,
                    qddot_full: qddot,
                }
            ),
            **play_kw
        )
        self.wait(duration=4)
        #
        self.play(
            TransformMatchingTex(
                lines[2].copy(), lines[3],
                key_map={
                    mass_matrix_full: mass_matrix,
                    qddot_full: qddot,
                }
            ),
            **play_kw
        )
        self.wait(duration=4)

# See https://github.com/3b1b/videos for many, many more