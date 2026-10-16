
# Carrega os pesos já treinados
model = GPTLanguageModel().to(device)
model.load_state_dict(
    torch.load("shakegpt.pth", map_location=device, weights_only=True)
)
model.eval()

print("Modelo treinado carregado!")
print("Gerando Shakespeare infinitamente...")
print("Arquivo: shakespeare_infinito.txt")
print("Pressione Ctrl+C para parar.")

import time

context = torch.zeros((1, 1), dtype=torch.long, device=device)
total_chars = 0
chunk_size = 1000

try:
    with open("shakespeare_infinito.txt", "a", encoding="utf-8") as f:
        while True:
            with torch.no_grad():
                generated = model.generate(
                    context,
                    max_new_tokens=chunk_size
                )

            new_tokens = generated[0, context.shape[1]:].tolist()
            new_text = decode(new_tokens)

            f.write(new_text)
            f.flush()

            total_chars += len(new_text)
            print(
                f"[{time.strftime('%H:%M:%S')}] "
                f"+{len(new_text)} caracteres | "
                f"Total nesta execução: {total_chars:,}"
            )

            context = generated[:, -block_size:]

except KeyboardInterrupt:
    print("\nGeração interrompida!")
    print(f"Total gerado: {total_chars:,} caracteres")
