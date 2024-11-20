const std = @import("std");
const Allocator = std.mem.Allocator;

pub const ImageHeader = extern struct {
    magic: u32,
    num_images: u32,
    rows: u32,
    columns: u32,
};

pub const LabelHeader = extern struct {
    magic: u32,
    num_items: u32,
};

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa_state.deinit();
    const gpa = gpa_state.allocator();

    const raw_label_file_data, const expected = try loadLabelData(gpa, "train-labels.idx1-ubyte");
    defer gpa.free(raw_label_file_data);
    defer gpa.free(expected);

    const test_label_file_data, const test_expected = try loadLabelData(gpa, "t10k-labels.idx1-ubyte");
    defer gpa.free(test_label_file_data);

    // Don't need it
    gpa.free(test_expected);

    var prng = std.Random.DefaultPrng.init(0);
    const rand = prng.random();

    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const input_data = try loadInputData(arena, "train-images.idx3-ubyte");
    const test_input_data = try loadInputData(arena, "t10k-images.idx3-ubyte");

    const num_samples = 60_000;
    const num_test_samples = 10_000;
    const input_size = 28 * 28;
    const hidden_size = 124;
    const output_size = 10;

    // Xavier initialization [https://365datascience.com/tutorials/machine-learning-tutorials/what-is-xavier-initialization/]
    const w1 = try arena.alignedAlloc(f32, 32, input_size * hidden_size);
    const w2 = try arena.alignedAlloc(f32, 32, hidden_size * output_size);
    const weight_scale1 = @sqrt(6.0 / @as(f32, @floatFromInt(input_size + hidden_size)));
    const weight_scale2 = @sqrt(6.0 / @as(f32, @floatFromInt(hidden_size + output_size)));

    for (w1) |*w| w.* = (rand.float(f32) * 2.0 - 1.0) * weight_scale1;
    for (w2) |*w| w.* = (rand.float(f32) * 2.0 - 1.0) * weight_scale2;

    const b1 = try arena.alignedAlloc(f32, 32, hidden_size);
    const b2 = try arena.alignedAlloc(f32, 32, output_size);
    @memset(b1, 0);
    @memset(b2, 0);

    // Gradients storage
    const dw1 = try arena.alignedAlloc(f32, 32, input_size * hidden_size);
    const dw2 = try arena.alignedAlloc(f32, 32, hidden_size * output_size);
    const db1 = try arena.alignedAlloc(f32, 32, hidden_size);
    const db2 = try arena.alignedAlloc(f32, 32, output_size);

    // Activations
    const hidden_activations = try arena.alignedAlloc(f32, 32, hidden_size * num_samples);
    const output_activations = try arena.alignedAlloc(f32, 32, output_size * num_samples);
    const test_output_activations = try arena.alignedAlloc(f32, 32, output_size * num_test_samples);

    // Batch storage
    const batch_size = 32;
    const batch_input = try arena.alignedAlloc(f32, 32, input_size * batch_size);
    const batch_hidden = try arena.alignedAlloc(f32, 32, hidden_size * batch_size);
    const batch_output = try arena.alignedAlloc(f32, 32, output_size * batch_size);
    const batch_expected = try arena.alignedAlloc(f32, 32, output_size * batch_size);

    feedforward(w1, w2, b1, b2, input_data, hidden_activations, output_activations, num_samples);
    var correct = countCorrectGuesses(output_activations, raw_label_file_data);
    var percent = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(num_samples));
    std.debug.print("Initial training accuracy: {d}/{d} ({d:.2}%)\n", .{ correct, num_samples, percent * 100 });

    feedforward(w1, w2, b1, b2, test_input_data, hidden_activations, test_output_activations, num_test_samples);
    correct = countCorrectGuesses(test_output_activations, test_label_file_data);
    percent = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(num_test_samples));
    std.debug.print("Initial test accuracy: {d}/{d} ({d:.2}%)\n", .{ correct, num_test_samples, percent * 100 });

    const initial_learning_rate = 0.08;
    const num_epochs = 55;

    // Training
    for (0..num_epochs) |epoch| {
        const learning_rate = initial_learning_rate / (1.0 + 0.1 * @as(f32, @floatFromInt(epoch)));

        // Randomize batch indices.
        var indices = try arena.alloc(u32, num_samples);
        for (indices, 0..) |*idx, i| idx.* = @intCast(i);
        var i: u32 = num_samples - 1;
        while (i > 0) : (i -= 1) {
            const j = rand.uintLessThan(u32, i + 1);
            const temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }

        var idx: u32 = 0;
        while (idx < num_samples) : (idx += batch_size) {
            const actual_batch_size = @min(batch_size, num_samples - idx);

            @memset(dw1, 0);
            @memset(dw2, 0);
            @memset(db1, 0);
            @memset(db2, 0);

            // Select batch input
            for (0..actual_batch_size) |j| {
                const sample_idx = indices[idx + j];
                @memcpy(batch_input[j * input_size .. (j + 1) * input_size], input_data[sample_idx * input_size .. (sample_idx + 1) * input_size]);
                @memcpy(batch_expected[j * output_size .. (j + 1) * output_size], expected[sample_idx * output_size .. (sample_idx + 1) * output_size]);
            }

            feedforward(w1, w2, b1, b2, batch_input, batch_hidden, batch_output, actual_batch_size);
            backprop(batch_input, batch_hidden, batch_output, batch_expected, w2, dw1, dw2, db1, db2, actual_batch_size);
            applyGradients(w1, w2, b1, b2, dw1, dw2, db1, db2, learning_rate);
        }

        feedforward(w1, w2, b1, b2, input_data, hidden_activations, output_activations, num_samples);
        correct = countCorrectGuesses(output_activations, raw_label_file_data);
        percent = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(num_samples));
        std.debug.print("Epoch {d}: Accuracy = {d}/{d} ({d:.2}%)\n", .{ epoch + 1, correct, num_samples, percent * 100 });

        feedforward(w1, w2, b1, b2, test_input_data, hidden_activations, test_output_activations, num_test_samples);
        correct = countCorrectGuesses(test_output_activations, test_label_file_data);
        percent = @as(f32, @floatFromInt(correct)) / @as(f32, @floatFromInt(num_test_samples));
        std.debug.print("Epoch {d}: Test accuracy = {d}/{d} ({d:.2}%)\n", .{ epoch + 1, correct, num_test_samples, percent * 100 });
    }

    // Save network weights
    {
        const weights_file = try std.fs.cwd().createFile("network1.weights", .{});
        defer weights_file.close();
        var bw = std.io.bufferedWriter(weights_file.writer());
        const writer = bw.writer();

        try writer.writeInt(u32, input_size, .little);
        try writer.writeInt(u32, 2, .little);
        try writer.writeInt(u32, hidden_size, .little);
        try writer.writeInt(u32, output_size, .little);

        for (w1) |w| {
            _ = try writer.write(std.mem.asBytes(&w)[0..4]);
        }

        for (w2) |w| {
            _ = try writer.write(std.mem.asBytes(&w)[0..4]);
        }

        try bw.flush();
    }

    // Save network biases
    {
        const weights_file = try std.fs.cwd().createFile("network1.biases", .{});
        defer weights_file.close();
        var bw = std.io.bufferedWriter(weights_file.writer());
        const writer = bw.writer();

        try writer.writeInt(u32, 2, .little);
        try writer.writeInt(u32, hidden_size, .little);
        try writer.writeInt(u32, output_size, .little);

        for (b1) |b| {
            _ = try writer.write(std.mem.asBytes(&b)[0..4]);
        }

        for (b2) |b| {
            _ = try writer.write(std.mem.asBytes(&b)[0..4]);
        }

        try bw.flush();
    }
}

pub fn feedforward(
    w1: []const f32,
    w2: []const f32,
    b1: []const f32,
    b2: []const f32,
    input: []const f32,
    hidden: []f32,
    output: []f32,
    num_samples: u32,
) void {
    const input_size = 28 * 28;
    const hidden_size = 124;
    const output_size = 10;

    // Input to hidden layer.
    for (0..num_samples) |sample_idx| {
        const hidden_slice = hidden[sample_idx * hidden_size .. (sample_idx + 1) * hidden_size];
        const input_slice = input[sample_idx * input_size .. (sample_idx + 1) * input_size];

        for (hidden_slice, 0..) |*h, i| {
            var sum: f32 = 0.0;
            for (input_slice, w1[i * input_size .. (i + 1) * input_size]) |in, w| {
                sum += in * w;
            }
            h.* = @max(0.0, sum + b1[i]);
        }
    }

    // Hidden to output layer
    for (0..num_samples) |sample_idx| {
        const output_slice = output[sample_idx * output_size .. (sample_idx + 1) * output_size];
        const hidden_slice = hidden[sample_idx * hidden_size .. (sample_idx + 1) * hidden_size];

        var max_val: f32 = -std.math.inf(f32);
        for (output_slice, 0..) |*o, i| {
            var sum: f32 = 0.0;
            for (hidden_slice, w2[i * hidden_size .. (i + 1) * hidden_size]) |h, w| {
                sum += h * w;
            }
            o.* = sum + b2[i];
            max_val = @max(max_val, o.*);
        }

        // softmax
        var sum_exp: f32 = 0.0;
        for (output_slice) |*o| {
            o.* = @exp(o.* - max_val);
            sum_exp += o.*;
        }

        for (output_slice) |*o| {
            o.* /= sum_exp;
        }
    }
}

fn backprop(
    input: []const f32,
    hidden: []const f32,
    output: []const f32,
    expected: []const f32,
    w2: []const f32,
    dw1: []f32,
    dw2: []f32,
    db1: []f32,
    db2: []f32,
    batch_size: u32,
) void {
    const input_size = 28 * 28;
    const hidden_size = 124;
    const output_size = 10;

    var delta_hidden: [hidden_size]f32 = undefined;

    for (0..batch_size) |sample_idx| {
        const in = input[sample_idx * input_size .. (sample_idx + 1) * input_size];
        const h = hidden[sample_idx * hidden_size .. (sample_idx + 1) * hidden_size];
        const o = output[sample_idx * output_size .. (sample_idx + 1) * output_size];
        const e = expected[sample_idx * output_size .. (sample_idx + 1) * output_size];

        var delta_output: [10]f32 = undefined;
        for (0..output_size) |i| {
            delta_output[i] = (o[i] - e[i]) / @as(f32, @floatFromInt(batch_size));
        }

        for (0..hidden_size) |i| {
            var sum: f32 = 0.0;
            for (0..output_size) |j| {
                sum += delta_output[j] * w2[j * hidden_size + i];
            }
            delta_hidden[i] = if (h[i] > 0.0) sum else 0.0;
        }

        for (0..output_size) |i| {
            for (0..hidden_size) |j| {
                dw2[i * hidden_size + j] += delta_output[i] * h[j];
            }
            db2[i] += delta_output[i];
        }

        for (0..hidden_size) |i| {
            for (0..input_size) |j| {
                dw1[i * input_size + j] += delta_hidden[i] * in[j];
            }
            db1[i] += delta_hidden[i];
        }
    }
}

fn applyGradients(
    w1: []f32,
    w2: []f32,
    b1: []f32,
    b2: []f32,
    dw1: []const f32,
    dw2: []const f32,
    db1: []const f32,
    db2: []const f32,
    learning_rate: f32,
) void {
    for (w1, dw1) |*w, dw| {
        w.* -= dw * learning_rate;
    }
    for (w2, dw2) |*w, dw| {
        w.* -= dw * learning_rate;
    }
    for (b1, db1) |*b, db| {
        b.* -= db * learning_rate;
    }

    for (b2, db2) |*b, db| {
        b.* -= db * learning_rate;
    }
}

pub fn countCorrectGuesses(activations: []const f32, labels: []const u8) u32 {
    var correct: u32 = 0;

    for (labels, 0..) |label, idx| {
        var max: f32 = -std.math.inf(f32);
        var max_idx: u8 = 0;
        for (0..10) |i| {
            const val = activations[idx * 10 + i];
            if (val > max) {
                max = val;
                max_idx = @intCast(i);
            }
        }

        if (max_idx == label) {
            correct += 1;
        }
    }

    return correct;
}

pub fn loadInputData(allocator: Allocator, path: []const u8) ![]f32 {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const header = try file.reader().readStructEndian(ImageHeader, .big);
    const raw = try file.readToEndAlloc(allocator, 1_000_000_000);
    defer allocator.free(raw);

    const input = try allocator.alloc(f32, header.num_images * header.columns * header.rows);
    for (raw, input) |r, *i| {
        i.* = @as(f32, @floatFromInt(r)) / 255.0;
    }

    return input;
}

pub fn loadLabelData(allocator: Allocator, path: []const u8) !struct { []u8, []f32 } {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    const header = try file.reader().readStructEndian(LabelHeader, .big);
    const raw = try file.readToEndAlloc(allocator, 1_000_000_000);

    const target = try allocator.alloc(f32, header.num_items * 10);
    @memset(target, 0);

    for (raw, 0..) |label, i| {
        target[10 * i + label] = 1.0;
    }

    return .{ raw, target };
}
