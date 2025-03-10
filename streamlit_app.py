import streamlit as st
import numpy as np
import io
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import openai
import logging

st.set_page_config(page_title="image-generator-demo", page_icon=":shark:", layout="wide")
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
        logging.FileHandler("app_errors.log"),  # 파일에 저장
        logging.StreamHandler()  # 터미널에 출력
    ])

def configure_sidebar() -> None:
    
    with st.sidebar:
        st.title("Image Generator Demo")

        st.subheader("🔑 Enter OpenAI API Key")
        openai_api_key = st.text_input("API Key", type="password")
        
        with st.expander(":gear: Settings"):
            st.write("Settings go here")
            # 이미지 크기 선택 (OpenAI DALL·E 지원 사이즈)
            image_size = st.radio("Select Image Size", ["256x256", "512x512", "1024x1024"], index=0)
            num_outputs = st.slider("Number of outputs", min_value=1, max_value=3, value=3)

    return image_size, num_outputs, openai_api_key

# OpenAI GPT를 사용하여 한글 프롬프트를 최적화된 영어 프롬프트로 변환하는 함수
def optimize_prompt_with_gpt(prompt, api_key):
    try:
        client = openai.OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant that improves text prompts for AI image generation."},
                {"role": "user", "content": f"Please refine and translate the following text into an optimized English prompt for AI image generation: {prompt}"}
            ],
            temperature=0.7,
            max_tokens=100
        )

        optimized_prompt = response.choices[0].message.content.strip()
        return optimized_prompt

    except Exception as e:
        st.error("❌ Prompt optimization failed.")
        st.error(str(e))
        return prompt  # 실패 시 원본 그대로 사용

def generate_ai_image_from_drawing(image_data, api_key, num_outputs, image_size):
    if not api_key:
        st.warning("⚠️ Please enter your OpenAI API Key!")
        return None

    if image_data is None:
        st.warning("⚠️ Please draw something on the canvas first!")
        return None

    st.write("⏳ Start Processing your drawing with AI...")

    # 이미지 변환
    # image = Image.fromarray((image_data * 255).astype(np.uint8))
    st.image(image_data, caption="Your Drawing")

    st.write("⏳ Uploading your drawing to AI...")

    # 최신 OpenAI SDK 방식 적용
    client = openai.OpenAI(api_key=api_key)

    response = client.images.generate(
        model="dall-e-2",  
        prompt="Generate an image by using the given sketch as a reference. The AI should complete and enhance the sketch while keeping its original style, shapes, and details. The output should be visually similar to the uploaded sketch, but with better color, shading, and fine details.",
        n=num_outputs,
        size=image_size
    )

    st.write("✅ AI image generated successfully!")

    return response


def main_page(image_size: str , num_outputs: int, openai_api_key: str) -> None:
    option = st.radio("Choose an input method:", ("Draw an image", "Describe with text"))

    client = openai.OpenAI(api_key=openai_api_key)

    # --- 그림 그리기 기능 ---
    if option == "Draw an image":
        st.subheader("🖌️ Draw your idea on the canvas")

        # 캔버스 생성
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",  # 배경 투명
            stroke_width=5,
            stroke_color="black",
            background_color="white",
            height=300,
            width=500,
            drawing_mode="freedraw",
            key="canvas",
        )

        # AI 이미지 수정 버튼
        if st.button("🎨 Generate Image from Drawing"):
            if canvas_result.image_data is not None:
                response = generate_ai_image_from_drawing(
                    canvas_result.image_data,
                    openai_api_key,
                    num_outputs,
                    image_size
                )

                if response:
                    # 이미지 개수에 따라 동적으로 컬럼 생성
                    cols = st.columns(num_outputs)
                    for idx, img in enumerate(response.data):
                        with cols[idx]:
                            st.image(img.url, caption=f"Generated Image {idx+1}")
            else:
                st.warning("⚠️ Please draw something on the canvas first!")
                

    # --- 텍스트 설명을 통한 이미지 생성 ---
    elif option == "Describe with text":
        st.subheader("📝 Describe the image you want AI to generate")
            
        prompt = st.text_area(
                    ":orange[**Enter prompt: start typing ✍🏾**]",
                    value="A vibrant and dynamic digital illustration of multiple people participating in an online challenge through a mobile app. The participants are actively engaged, checking their rankings on a leaderboard within the app, and striving to improve their performance. The image conveys a sense of competition, motivation, and teamwork. The background features a futuristic interface with charts, rankings, and progress bars, emphasizing the challenge dynamics. The overall atmosphere is energetic and inspiring, with users pushing themselves to achieve higher scores and celebrating their progress together.")  


        if st.button("🎨 Generate Image"):
            try:
                if not openai_api_key:
                    st.warning("⚠️ Please enter your OpenAI API Key!")
                elif not prompt:
                    st.warning("⚠️ Please enter a description!")
                else:
                    # GPT를 통해 최적화된 프롬프트 변환
                    if prompt:
                        optimized_prompt = optimize_prompt_with_gpt(prompt, openai_api_key)
                        st.write(f"🔄 Optimized Prompt: **{optimized_prompt}**")

                    st.write("⏳ Generating AI image...")
                    openai.api_key = openai_api_key
                    response = client.images.generate(
                        model="dall-e-2",  # 최신 모델 지정
                        prompt=optimized_prompt,
                        n=num_outputs,
                        size=image_size
                    )
                    
                    # 이미지 개수에 따라 동적으로 컬럼 생성
                    cols = st.columns(num_outputs)
                    for idx, img in enumerate(response.data):
                        with cols[idx]:
                            st.image(img.url, caption=f"Generated Image {idx+1}")

            except Exception as e:
                logging.error(f"Error occurred: {e}")
                st.write("⚠️ An unexpected error occurred. Please try again later.")
                    


def main() -> None:
    image_size, num_outputs, openai_api_key = configure_sidebar()
    main_page(image_size, num_outputs, openai_api_key)

if __name__ == "__main__":
    main()

