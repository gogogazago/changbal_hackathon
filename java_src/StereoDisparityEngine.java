import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import javax.imageio.ImageIO;

public class StereoDisparityEngine {

    private static final float IPD_OFFSET = 0.032f; // 64mm IPD (양안 거리 절반 = 0.016f)
    private static final int GRID_SIZE = 64;       // 64x64 정밀 해상도

    public static void main(String[] args) {
        // 대상 오브젝트 설정
        String targetObject = "headphone";

        String rendersFolderDir = "pixel_frames/" + targetObject + "/3d_renders";
        String outputCppHeader = "pixel_frames/" + targetObject + "/3d_renders/stereo_data.h";

        File rendersFolder = new File(rendersFolderDir);
        // 폴더 내의 SBS 입체 프리뷰 이미지(_3d.png) 파일들을 검색합니다.
        File[] renderFiles = rendersFolder.listFiles((dir, name) -> name.endsWith("_3d.png"));

        if (renderFiles == null || renderFiles.length == 0) {
            System.err.println("❌ SBS 3D 렌더 이미지를 찾을 수 없습니다: " + rendersFolderDir);
            System.err.println("⚠️ 기존 단일 이미지 파이프라인으로 전환하거나 경로를 확인하세요.");
            return;
        }
        Arrays.sort(renderFiles);

        try (FileWriter writer = new FileWriter(outputCppHeader)) {
            System.out.println("🚀 [Stereo Disparity Engine] SBS 입체 이미지를 이용한 정밀 3D 좌표 복원 시작...");

            writer.write("#ifndef STEREO_DATA_H\n#define STEREO_DATA_H\n\n");
            writer.write("int TOTAL_FRAMES = " + renderFiles.length + ";\n");
            writer.write("int POINTS_PER_FRAME = " + (GRID_SIZE * GRID_SIZE) + ";\n\n");
            writer.write("struct StereoPoint {\n    float lx, ly, lz;\n    float rx, ry, rz;\n    float r, g, b;\n};\n\n");

            for (int f = 0; f < renderFiles.length; f++) {
                File renderFile = renderFiles[f];
                String name = renderFile.getName();

                // 프레임 번호 추출 (예: frame_0000_3d.png -> 0000)
                String frameNum = name.replaceAll("[^0-9]", "");
                if (frameNum.length() > 4) frameNum = frameNum.substring(0, 4);

                System.out.println("📦 [시차 정밀 분석 중] Frame " + frameNum + " ➡️ Left/Right 이미지 분리 및 3D 매핑");
                writer.write("inline static StereoPoint frame_" + frameNum + "[" + (GRID_SIZE * GRID_SIZE) + "] = {\n");

                // 1. SBS 이미지 읽기
                BufferedImage sbsImg = ImageIO.read(renderFile);
                int width = sbsImg.getWidth();
                int height = sbsImg.getHeight();
                int midX = width / 2;

                // 2. 왼쪽 눈 이미지와 오른쪽 눈 이미지를 정확히 반으로 쪼갭니다 (중앙 분할선 제외 마진 적용)
                int safetyMargin = 2; // 중앙 회색 분리선 번짐을 막기 위한 여백
                BufferedImage leftImg = sbsImg.getSubimage(0, 0, midX - safetyMargin, height);
                BufferedImage rightImg = sbsImg.getSubimage(midX + safetyMargin, 0, width - midX - safetyMargin, height);

                // 3. VR 렌더링에 적합한 64x64 사이즈로 각각 해상도 매핑
                BufferedImage scaledLeft = scaleImage(leftImg, GRID_SIZE, GRID_SIZE);
                BufferedImage scaledRight = scaleImage(rightImg, GRID_SIZE, GRID_SIZE);

                for (int y = 0; y < GRID_SIZE; y++) {
                    for (int x = 0; x < GRID_SIZE; x++) {
                        // OpenGL 정규화 좌표계 (-1.0f ~ +1.0f) 적용
                        float posX = ((x - (GRID_SIZE / 2.0f)) / (GRID_SIZE / 2.0f)) * 1.0f;
                        float posY = (((GRID_SIZE / 2.0f) - y) / (GRID_SIZE / 2.0f)) * 1.0f;

                        // 좌측 픽셀 정보 획득
                        Color colorL = new Color(scaledLeft.getRGB(x, y));
                        float rL = colorL.getRed() / 255.0f;
                        float gL = colorL.getGreen() / 255.0f;
                        float bL = colorL.getBlue() / 255.0f;

                        float posZ;
                        float lx = posX;
                        float rx = posX;

                        // 배경 검정색 픽셀 필터링 (렌더링 차단)
                        if (rL < 0.05f && gL < 0.05f && bL < 0.05f) {
                            posZ = -100.0f; // 시야 밖으로 날려 숨김
                        } else {
                            // ★ 핵심: 양안 시차(Stereo Disparity) 추적 알고리즘
                            // 좌측 픽셀과 가장 유사한 우측 픽셀의 수평 위치 차이(d)를 구합니다.
                            int maxSearchRange = 12; // 수평 탐색 범위 (픽셀)
                            int bestDisparity = 0;
                            float minColorDiff = Float.MAX_VALUE;

                            for (int d = 0; d < maxSearchRange; d++) {
                                int rxTarget = x - d; // 우측 눈의 상은 좌측 눈보다 안쪽에 맺힙니다.
                                if (rxTarget < 0) break;

                                Color colorR = new Color(scaledRight.getRGB(rxTarget, y));
                                float diff = Math.abs(colorL.getRed() - colorR.getRed()) +
                                        Math.abs(colorL.getGreen() - colorR.getGreen()) +
                                        Math.abs(colorL.getBlue() - colorR.getBlue());

                                if (diff < minColorDiff) {
                                    minColorDiff = diff;
                                    bestDisparity = d;
                                }
                            }

                            // 찾아낸 시차(Disparity)를 이용해 물리적 깊이(Z축) 계산!
                            // 시차가 클수록(가까이 있을수록) 깊이가 튀어나오고, 시차가 작을수록(멀리 있을수록) 깊이가 평평해집니다.
                            float disparityRatio = (float) bestDisparity / maxSearchRange;
                            posZ = 0.2f + (disparityRatio * 0.6f); // 깊이 가속 및 보정

                            // 양안 오프셋 적용
                            lx = posX + (IPD_OFFSET * 0.5f);
                            rx = posX - (IPD_OFFSET * 0.5f);
                        }

                        // C++ 정점 데이터로 변환 (Locale.US 적용으로 콤마 오버플로우 방지)
                        writer.write(String.format(Locale.US,
                                "    {%.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff, %.4ff},\n",
                                lx, posY, -posZ, rx, posY, -posZ, rL, gL, bL));
                    }
                }
                writer.write("};\n\n");
            }

            writer.write("#endif // STEREO_DATA_H\n");
            System.out.println("🏆 [완벽 복원] SBS 시차 추적을 반영한 최종 헤더 파일 생성 완료!");

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