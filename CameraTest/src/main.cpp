#include <librealsense2/rs.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>

int main() try {
    // Start depth-only stream
    rs2::pipeline pipe;
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
    pipe.start(cfg);

    // (Optional) small warm-up so auto-calibration settles
    for (int i = 0; i < 10; ++i) pipe.wait_for_frames();

    const int R = 5; // radius for averaging window -> (2R+1)^2 = 11x11
    std::cout << std::fixed << std::setprecision(3);

    while (true) {
        rs2::frameset fs = pipe.wait_for_frames();
        rs2::depth_frame depth = fs.get_depth_frame();

        int w = depth.get_width();
        int h = depth.get_height();
        int cx = w / 2;
        int cy = h / 2;

        double sum = 0.0;
        int count = 0;

        // Average a small ROI around the center for a steadier reading
        for (int dy = -R; dy <= R; ++dy) {
            int y = cy + dy;
            if (y < 0 || y >= h) continue;
            for (int dx = -R; dx <= R; ++dx) {
                int x = cx + dx;
                if (x < 0 || x >= w) continue;
                float z = depth.get_distance(x, y); // meters; 0 means invalid
                if (z > 0.f) { sum += z; ++count; }
            }
        }

        double dist_m = (count ? sum / count : 0.0);
        std::cout << "Center distance: " << dist_m << " m" << std::endl;

        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

} catch (const rs2::error& e) {
    std::cerr << "RealSense error: " << e.what()
              << " (" << e.get_failed_function() << ")\n";
    return 1;
} catch (const std::exception& e) {
    std::cerr << "std::exception: " << e.what() << "\n";
    return 1;
}
