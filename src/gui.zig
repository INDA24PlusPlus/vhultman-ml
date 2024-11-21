const std = @import("std");
const glfw = @import("zglfw");
const zgui = @import("zgui");
const zopengl = @import("zopengl");
const ml = @import("train.zig");

const ImageHeader = extern struct {
    magic: u32,
    num_images: u32,
    rows: u32,
    columns: u32,
};

const LabelHeader = extern struct {
    magic: u32,
    num_items: u32,
};

const Global = struct {
    var image_data: []u8 = undefined;
    var selected_image: u32 = 0;
    var correct_train_samples: u32 = 0;
    var correct_test_samples: u32 = 0;
};

pub fn main() !void {
    try glfw.init();
    defer glfw.terminate();

    const gl_major = 4;
    const gl_minor = 0;
    glfw.windowHintTyped(.context_version_major, gl_major);
    glfw.windowHintTyped(.context_version_minor, gl_minor);
    glfw.windowHintTyped(.opengl_profile, .opengl_core_profile);
    glfw.windowHintTyped(.opengl_forward_compat, true);
    glfw.windowHintTyped(.client_api, .opengl_api);
    glfw.windowHintTyped(.doublebuffer, true);
    glfw.windowHintTyped(.floating, true);

    const window = try glfw.Window.create(1920, 1080, "ML", null);
    defer window.destroy();
    window.setSizeLimits(400, 400, -1, -1);

    glfw.makeContextCurrent(window);
    glfw.swapInterval(1);
    _ = window.setScrollCallback(scrollCallback);
    _ = window.setCursorPosCallback(mousePosCallback);

    try zopengl.loadCoreProfile(glfw.getProcAddress, gl_major, gl_minor);
    const gl = zopengl.bindings;

    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    zgui.init(gpa);
    defer zgui.deinit();

    const scale_factor = scale_factor: {
        const scale = window.getContentScale();
        break :scale_factor @max(scale[0], scale[1]);
    };
    _ = zgui.io.addFontFromFile(
        "assets/Roboto-Medium.ttf",
        std.math.floor(16.0 * scale_factor),
    );

    zgui.getStyle().scaleAllSizes(scale_factor);
    zgui.io.setConfigFlags(.{
        .dock_enable = true,
    });

    zgui.backend.init(window);
    defer zgui.backend.deinit();

    var show_demo_window = true;

    const texture_handle = createTexture();

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const raw_label_file_data, const expected = try ml.loadLabelData(gpa, "train-labels.idx1-ubyte");
    gpa.free(expected);
    defer gpa.free(raw_label_file_data);

    const test_label_file_data, const test_expected = try ml.loadLabelData(gpa, "t10k-labels.idx1-ubyte");
    gpa.free(test_expected);
    defer gpa.free(test_label_file_data);

    const image_file = try std.fs.cwd().openFile("train-images.idx3-ubyte", .{});
    defer image_file.close();
    const header = try image_file.reader().readStructEndian(ml.ImageHeader, .big);

    const num_samples = 60_000;
    Global.image_data = try arena.alloc(u8, header.columns * header.rows * header.num_images);
    const input_data = try arena.alloc(f32, header.columns * header.rows * header.num_images);
    {
        const raw_image_data = try image_file.readToEndAlloc(gpa, 1_000_000_000);
        defer gpa.free(raw_image_data);

        for (Global.image_data, input_data, raw_image_data) |*i, *in, c| {
            i.* = @intCast(c);
            in.* = @as(f32, @floatFromInt(c)) / 255.0;
        }
    }

    const test_input_data = try ml.loadInputData(gpa, "t10k-images.idx3-ubyte");
    defer gpa.free(test_input_data);

    const num_neurons = 124;
    const weights = blk: {
        const file = try std.fs.cwd().openFile("98-06.weights", .{});
        defer file.close();

        const weights = try arena.alignedAlloc(f32, 32, 28 * 28 * num_neurons + num_neurons * 10);

        // input size.
        std.debug.assert(try file.reader().readInt(u32, .little) == 28 * 28);

        // layer count.
        std.debug.assert(try file.reader().readInt(u32, .little) == 2);
        // layer size
        std.debug.assert(try file.reader().readInt(u32, .little) == num_neurons);
        std.debug.assert(try file.reader().readInt(u32, .little) == 10);

        const ptr = @as([*]u8, @ptrCast(@alignCast(weights.ptr)))[0 .. 4 * weights.len];
        _ = try file.reader().readAtLeast(ptr[0 .. 4 * 28 * 28 * num_neurons], 4 * 28 * 28 * num_neurons);
        _ = try file.reader().readAtLeast(ptr[4 * 28 * 28 * num_neurons ..], 4 * num_neurons * 10);

        break :blk weights;
    };

    const biases = blk: {
        const file = try std.fs.cwd().openFile("98-06.biases", .{});
        defer file.close();

        const biases = try arena.alignedAlloc(f32, 32, num_neurons + 10);

        // layer count.
        std.debug.assert(try file.reader().readInt(u32, .little) == 2);
        // layer size
        std.debug.assert(try file.reader().readInt(u32, .little) == num_neurons);
        std.debug.assert(try file.reader().readInt(u32, .little) == 10);

        const ptr = @as([*]u8, @ptrCast(@alignCast(biases.ptr)))[0 .. biases.len * 4];
        _ = try file.reader().readAtLeast(ptr[0..num_neurons], num_neurons);
        _ = try file.reader().readAtLeast(ptr[num_neurons..], 10);

        break :blk biases;
    };

    const w1 = weights[0 .. num_neurons * 28 * 28];
    const w2 = weights[num_neurons * 28 * 28 ..];
    const b1 = biases[0..num_neurons];
    const b2 = biases[num_neurons..];

    const output = try arena.alloc(f32, 10 * num_samples);
    const hidden_out = try arena.alloc(f32, num_neurons * num_samples);

    ml.feedforward(w1, w2, b1, b2, test_input_data, hidden_out, output, 10_000);
    Global.correct_test_samples = ml.countCorrectGuesses(output, test_label_file_data);

    ml.feedforward(w1, w2, b1, b2, input_data, hidden_out, output, num_samples);
    Global.correct_train_samples = ml.countCorrectGuesses(output, raw_label_file_data);

    while (!window.shouldClose() and window.getKey(.escape) != .press) {
        glfw.pollEvents();

        gl.clearBufferfv(gl.COLOR, 0, &[_]f32{ 0.3, 0.5, 0.3, 1.0 });

        const fb_size = window.getFramebufferSize();
        zgui.backend.newFrame(@intCast(fb_size[0]), @intCast(fb_size[1]));

        const viewport = zgui.getMainViewport();
        _ = zgui.DockSpaceOverViewport(0, viewport, .{});

        const si = Global.selected_image;
        drawImageView(texture_handle, header.num_images);
        const current_image_slice = Global.image_data[28 * 28 * si .. 28 * 28 * (si + 1)];
        const hidden_activation = hidden_out[num_neurons * si .. num_neurons * (si + 1)];
        const output_activaiton = output[10 * si .. 10 * (si + 1)];

        drawNetwork(window, current_image_slice, &.{ hidden_activation, output_activaiton }, &.{ w1, w2 }, &.{ b1, b2 });
        zgui.showDemoWindow(&show_demo_window);

        zgui.backend.draw();
        window.swapBuffers();
    }
}

const PanSettings = struct {
    var window_offset: [2]f32 = .{ 0, 0 };
    var zoom: f32 = 1.0;
    var pan: [2]f32 = .{ 0, 0 };
    var last_mouse_pos: [2]f32 = .{ 0, 0 };
    var is_panning = false;
};

fn mousePosCallback(window: *glfw.Window, xpos: f64, ypos: f64) callconv(.C) void {
    const x: f32 = @floatCast(xpos);
    const y: f32 = @floatCast(ypos);
    if (window.getMouseButton(.left) == .press) {
        if (!PanSettings.is_panning) {
            PanSettings.is_panning = true;
            PanSettings.last_mouse_pos = .{ x, y };
        }

        PanSettings.pan[0] -= (x - PanSettings.last_mouse_pos[0]) / PanSettings.zoom;
        PanSettings.pan[1] -= (y - PanSettings.last_mouse_pos[1]) / PanSettings.zoom;
        PanSettings.last_mouse_pos = .{ x, y };
    } else {
        PanSettings.is_panning = false;
    }
}

fn scrollCallback(window: *glfw.Window, _: f64, yoffset: f64) callconv(.C) void {
    const zoom_speed = 0.1;
    const mouse_pos_64 = window.getCursorPos();
    const mouse_pos: [2]f32 = .{
        @floatCast(mouse_pos_64[0]),
        @floatCast(mouse_pos_64[1]),
    };

    const before_zoom = screenToWorld(mouse_pos, PanSettings.pan, PanSettings.zoom);
    PanSettings.zoom *= 1.0 + @as(f32, @floatCast(yoffset)) * zoom_speed;
    PanSettings.zoom = @max(0.1, PanSettings.zoom);
    const after_zoom = screenToWorld(mouse_pos, PanSettings.pan, PanSettings.zoom);

    PanSettings.pan[0] += (before_zoom[0] - after_zoom[0]);
    PanSettings.pan[1] += (before_zoom[1] - after_zoom[1]);
}

fn worldToScreen(world: [2]f32, pan: [2]f32, zoom: f32) [2]f32 {
    return .{
        (world[0] - pan[0]) * zoom + PanSettings.window_offset[0],
        (world[1] - pan[1]) * zoom + PanSettings.window_offset[1],
    };
}

fn screenToWorld(screen: [2]f32, pan: [2]f32, zoom: f32) [2]f32 {
    return .{
        (screen[0] - PanSettings.window_offset[0]) / zoom + pan[0],
        (screen[1] - PanSettings.window_offset[1]) / zoom + pan[1],
    };
}

fn drawNetwork(
    window: *glfw.Window,
    input: []const u8,
    list_of_activations: []const []const f32,
    list_of_weights: []const []const f32,
    list_of_biases: []const []const f32,
) void {
    if (zgui.begin("Layer", .{})) {
        const window_offset = zgui.getCursorScreenPos();
        PanSettings.window_offset = window_offset;
        const draw_list = zgui.getWindowDrawList();

        const neuron_spacing = 16;
        const layer_spacing = 100;
        const radius = 8;

        const screen_mouse_pos = window.getCursorPos();
        const mouse_pos = screenToWorld(
            .{ @floatCast(screen_mouse_pos[0]), @floatCast(screen_mouse_pos[1]) },
            PanSettings.pan,
            PanSettings.zoom,
        );

        // draw input.
        for (0..28) |y| {
            for (0..28) |x| {
                const fx: f32 = @floatFromInt(x);
                const fy: f32 = @floatFromInt(y);
                const index = y * 28 + x;
                const activation = @as(f32, @floatFromInt(input[index])) / 255.0;

                const center = [2]f32{ neuron_spacing * fx, neuron_spacing * fy };
                const color = zgui.colorConvertFloat4ToU32(.{
                    activation,
                    activation,
                    activation,
                    1.0,
                });

                const dx = mouse_pos[0] - center[0];
                const dy = mouse_pos[1] - center[1];
                if (dx * dx + dy * dy <= radius * radius) {
                    if (zgui.beginTooltip()) {
                        zgui.text("{d:.2}", .{activation});
                    }
                    zgui.endTooltip();
                }

                const screen_pos = worldToScreen(center, PanSettings.pan, PanSettings.zoom);

                draw_list.addCircleFilled(.{
                    .p = screen_pos,
                    .col = color,
                    .r = radius * PanSettings.zoom,
                });
            }
        }

        // draw first layer.
        for (list_of_activations[0], list_of_biases[0], 0..) |activation, bias, idx| {
            const x = neuron_spacing * 28 + layer_spacing;
            const y = @as(f32, @floatFromInt(idx)) * neuron_spacing;
            const color = zgui.colorConvertFloat4ToU32(.{
                activation,
                activation,
                activation,
                1.0,
            });
            const screen_pos = worldToScreen(.{ x, y }, PanSettings.pan, PanSettings.zoom);

            const dx = mouse_pos[0] - x;
            const dy = mouse_pos[1] - y;
            if (dx * dx + dy * dy <= radius * radius) {
                if (zgui.beginTooltip()) {
                    zgui.text("Activation: {d:.2}", .{activation});
                    zgui.text("Bias: {d:.2}", .{bias});
                }
                zgui.endTooltip();

                // draw weight lines.
                for (0..28) |input_y| {
                    for (0..28) |input_x| {
                        const fx: f32 = @floatFromInt(input_x);
                        const fy: f32 = @floatFromInt(input_y);
                        const index = input_y * 28 + input_x;
                        const weight = (list_of_weights[0][idx * 28 * 28 + index] + 0.5) / 1.0;

                        const center = [2]f32{ neuron_spacing * fx, neuron_spacing * fy };
                        const weight_line_color = zgui.colorConvertFloat4ToU32(.{
                            weight,
                            weight,
                            weight,
                            1.0,
                        });

                        const input_pos = worldToScreen(center, PanSettings.pan, PanSettings.zoom);
                        {
                            const weight_vis_pos = worldToScreen(
                                .{ center[0], center[1] + 28 * neuron_spacing + layer_spacing },
                                PanSettings.pan,
                                PanSettings.zoom,
                            );

                            draw_list.addCircleFilled(.{
                                .p = weight_vis_pos,
                                .col = weight_line_color,
                                .r = radius * PanSettings.zoom,
                            });
                        }

                        draw_list.addLine(.{
                            .p1 = input_pos,
                            .p2 = screen_pos,
                            .col = weight_line_color,
                            .thickness = 3,
                        });
                    }
                }
            }

            draw_list.addCircleFilled(.{
                .p = screen_pos,
                .col = color,
                .r = radius * PanSettings.zoom,
            });
        }

        if (list_of_activations.len > 1) {
            for (list_of_activations[1..], list_of_biases[1..], 1..) |activations, biases, idx| {
                drawLayer(
                    activations,
                    list_of_weights[idx],
                    biases,
                    list_of_activations[idx - 1].len,
                    mouse_pos,
                    28 * neuron_spacing + (idx + 1) * layer_spacing,
                    0,
                );
            }
        }
    }

    zgui.end();
}

fn drawLayer(
    activations: []const f32,
    weights: []const f32,
    biases: []const f32,
    prev_layer_neuron_count: usize,
    mouse_pos: [2]f32,
    start_x: usize,
    start_y: usize,
) void {
    const draw_list = zgui.getWindowDrawList();
    const neuron_spacing = 16;
    const layer_spacing = 100;
    const radius = 8;

    for (activations, biases, 0..) |activation, bias, iy| {
        const x: f32 = @floatFromInt(start_x);
        const y: f32 = @floatFromInt(start_y + iy * neuron_spacing);

        const color = zgui.colorConvertFloat4ToU32(.{
            activation,
            activation,
            activation,
            1.0,
        });
        const screen_pos = worldToScreen(.{ x, y }, PanSettings.pan, PanSettings.zoom);

        const dx = mouse_pos[0] - x;
        const dy = mouse_pos[1] - y;
        if (dx * dx + dy * dy <= radius * radius) {
            if (zgui.beginTooltip()) {
                zgui.text("Activation: {d:.2}", .{activation});
                zgui.text("Bias: {d:.2}", .{bias});
            }
            zgui.endTooltip();

            // draw weight lines.
            for (0..prev_layer_neuron_count) |prev_y| {
                const fx: f32 = x - layer_spacing;
                const fy: f32 = @floatFromInt(neuron_spacing * prev_y);
                const weight = (weights[prev_y] + 0.5) / 1;

                const center = [2]f32{ fx, fy };
                const weight_line_color = zgui.colorConvertFloat4ToU32(.{
                    weight,
                    weight,
                    weight,
                    1.0,
                });

                const input_pos = worldToScreen(center, PanSettings.pan, PanSettings.zoom);

                draw_list.addLine(.{
                    .p1 = input_pos,
                    .p2 = screen_pos,
                    .col = weight_line_color,
                    .thickness = 3,
                });
            }
        }

        draw_list.addCircleFilled(.{
            .p = screen_pos,
            .col = color,
            .r = radius * PanSettings.zoom,
        });
    }
}

fn drawImageView(ident: zgui.TextureIdent, num_images: u32) void {
    var current_img: i32 = @intCast(Global.selected_image);
    if (zgui.begin("Digit", .{})) {
        zgui.image(ident, .{ .w = 256, .h = 256 });
        if (zgui.inputInt("Image", .{ .v = &current_img })) {
            current_img = std.math.clamp(current_img, 0, @as(i32, @intCast(num_images)));
            Global.selected_image = @intCast(current_img);
            const img_start = 28 * 28 * Global.selected_image;

            const gl = zopengl.bindings;
            gl.texImage2D(
                gl.TEXTURE_2D,
                0,
                gl.RED,
                28,
                28,
                0,
                gl.RED,
                gl.UNSIGNED_BYTE,
                Global.image_data[img_start .. img_start + 28 * 28].ptr,
            );
        }

        zgui.text("Training samples accuracy: {d}/{d} ({d:.2}%)", .{
            Global.correct_train_samples,
            60_000,
            @as(f32, @floatFromInt(Global.correct_train_samples)) / 60_000 * 100,
        });

        zgui.text("Test samples accuracy: {d}/{d} ({d:.2}%)", .{
            Global.correct_test_samples,
            10_000,
            @as(f32, @floatFromInt(Global.correct_test_samples)) / 10_000 * 100,
        });
    }
    zgui.end();
}

fn createTexture() zgui.TextureIdent {
    const gl = zopengl.bindings;
    var texture_ident: u32 = undefined;
    gl.genTextures(1, &texture_ident);
    gl.bindTexture(gl.TEXTURE_2D, texture_ident);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RED,
        @intCast(28),
        @intCast(28),
        0,
        gl.RED,
        gl.UNSIGNED_BYTE,
        Global.image_data.ptr,
    );
    return @ptrFromInt(texture_ident);
}
