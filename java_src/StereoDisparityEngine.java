import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import javax.imageio.ImageIO;

public class StereoDisparityEngine {

    private static final float IPD_OFFSET = 0.032f;
    private static final int GRID_SIZE = 64;

    public static void main(String[] args) {
        String targetObject = "headphone";

        String gridFolderDir = "pixel_frames/" + targetObject + "/pixel_grid";
        String depthFolderDir = "pixel_frames/" + targetObject + "/depth_maps";
        String outputCppHeader = "pixel_frames/" + targetObject + "/3d_renders/stereo_data.h";

        File gridFolder = new File(gridFolderDir);
        File[] gridFiles = gridFolder.listFiles((dir, name) -> name.endsWith(".png"));

        if (gridFiles == null || gridFiles.length == 0) {
            System.err.println("❌ 픽셀 그리드 이미지를 찾을 수 없습니다: " + gridFolderDir);
            return;
        }
        Arrays.sort(gridFiles);

        try (FileWriter writer = new FileWriter(outputCppHeader)) {
            System.out.println("🚀 [IntelliJ Java] 대용량 데이터 정밀 연산 및 C++ 추출 시작...");

            writer.write("#ifndef STEREO_DATA_H\n#define STEREO_DATA_H\n\n");
            writer.write("int TOTAL_FRAMES = " + gridFiles.length + ";\n");
            writer.write("int POINTS_PER_FRAME = " + (GRID_SIZE * GRID_SIZE) + ";\n\n");
            writer.write("struct StereoPoint {\n    float lx, ly, lz;\n    float rx, ry, rz;\n    float r, g, b;\n};\n\n");

            for (int f = 0; f < gridFiles.length; f++) {
                File gridFile = gridFiles[f];
                String name = gridFile.getName();

                // 파일명 매칭 유연화 (frame_0000_pixels.png -> 0000 추출)
                String frameNum = name.replaceAll("[^0-9]", "");
                if (frameNum.length() > 4) frameNum = frameNum.substring(0, 4);

                // depth_maps 폴더 스캔
                File depthFolder = new File(depthFolderDir);
                final String targetNum = frameNum;
                File[] matchingDepth = depthFolder.listFiles((dir, dName) -> dName.contains(targetNum) && dName.endsWith(".png"));

                if (matchingDepth == null || matchingDepth.length == 0) {
                    System.out.println("⚠️ 매칭되는 뎁스 파일을 찾지 못해 순서대로 매칭합니다.");
                    File[] allDepth = depthFolder.listFiles((dir, dName) -> dName.endsWith(".png"));
                    if (allDepth != null && allDepth.length > f) {
                        matchingDepth = new File[]{allDepth[f]};
                    } else {
                        continue;
                    }
                }

                System.out.println("📦 [연산 중] Frame " + frameNum + " ➡️ 데이터 물리 좌표 매핑");
                writer.write("inline static StereoPoint frame_" + frameNum + "[" + (GRID_SIZE * GRID_SIZE) + "] = {\n");

                BufferedImage colorImg = ImageIO.read(gridFile);
                BufferedImage depthImg = ImageIO.read(matchingDepth[0]);
                BufferedImage scaledColor = scaleImage(colorImg, GRID_SIZE, GRID_SIZE);
                BufferedImage scaledDepth = scaleImage(depthImg, GRID_SIZE, GRID_SIZE);

                for (int y = 0; y < GRID_SIZE; y++) {
                    for (int x = 0; x < GRID_SIZE; x++) {

                        // 1. OpenGL 표준 가상 공간에 맞춰 -1.0 ~ +1.0 스케일로 바나나 크기 조정
                        float posX = ((x - (GRID_SIZE / 2.0f)) / (GRID_SIZE / 2.0f)) * 1.0f;
                        float posY = (((GRID_SIZE / 2.0f) - y) / (GRID_SIZE / 2.0f)) * 1.0f;

                        Color color = new Color(scaledColor.getRGB(x, y));
                        float r = color.getRed() / 255.0f;
                        float g = color.getGreen() / 255.0f;
                        float b = color.getBlue() / 255.0f;

                        Color depthColor = new Color(scaledDepth.getRGB(x, y));
                        float grayDepth = (depthColor.getRed() * 0.299f + depthColor.getGreen() * 0.587f + depthColor.getBlue() * 0.114f) / 255.0f;

                        // 2. 렌더러 가시거리 내 기준점(0, 0, 0) 근처에 뎁스값 매핑
                        float posZ = 0.0f + (grayDepth * 0.5f);

                        // ★ 해결책 1: 배경 픽셀(블랙)이 바나나를 가리는 현상(Occlusion) 원천 차단
                        // 픽셀이 어두우면 렌더링 시야(Frustum) 바깥인 -100.0f로 던져버려서 아예 안 보이게 만듭니다.
                        if (r < 0.05f && g < 0.05f && b < 0.05f) {
                            posZ = -100.0f;
                        }

                        // 3. 64mm IPD 양안 시차 오프셋을 정밀 적용
                        float lx = posX + (IPD_OFFSET * 0.5f);
                        float rx = posX - (IPD_OFFSET * 0.5f);

                        // ★ 해결책 2: Locale.US 강제 적용 및 C++ float 명시(f)로 좌표 붕괴 방지
                        writer.write(String.format(java.util.Locale.US,
                                "    {%.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff},\n",
                                lx, posY, posZ, rx, posY, posZ, r, g, b));
                    }
                }
                writer.write("};\n\n");
            }

            writer.write("#endif // STEREO_DATA_H\n");
            System.out.println("🏆 [추출 성공] 진짜 데이터가 실린 헤더 파일 완성!");

        } catch (IOException e) {
            System.err.println("❌ 에러 발생: " + e.getMessage());
        }
    }

    private static BufferedImage scaleImage(BufferedImage original, int width, int height) {
        BufferedImage scaled = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        java.awt.Graphics2D g = scaled.createGraphics();
        g.setRenderingHint(java.awt.RenderingHints.KEY_INTERPOLATION, java.awt.RenderingHints.VALUE_INTERPOLATION_NEAREST_NEIGHBOR);
        g.drawImage(original, 0, 0, width, height, null);
        g.dispose();
        return scaled;
    }
}