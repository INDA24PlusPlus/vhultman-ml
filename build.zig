const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const train = b.option(bool, "train", "Build the application for training") orelse false;

    const exe = b.addExecutable(.{
        .name = "ml",
        .root_source_file = b.path(if (train) "src/train.zig" else "src/gui.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (!train) {
        const zopengl = b.dependency("zopengl", .{});
        exe.root_module.addImport("zopengl", zopengl.module("root"));

        const zglfw = b.dependency("zglfw", .{});
        exe.root_module.addImport("zglfw", zglfw.module("root"));
        exe.linkLibrary(zglfw.artifact("glfw"));

        const zgui = b.dependency("zgui", .{
            .shared = false,
            .with_implot = false,
            .backend = .glfw_opengl3,
        });
        exe.root_module.addImport("zgui", zgui.module("root"));
        exe.linkLibrary(zgui.artifact("imgui"));
    }

    b.installArtifact(exe);

    const run_cmd = b.addRunArtifact(exe);

    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);
}
