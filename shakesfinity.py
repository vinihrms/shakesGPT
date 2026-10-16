
# Geração infinita usando o modelo já treinado em memória
import time

model.eval()
context = torch.zeros((1, 1), dtype=torch.long, device=device)
total_chars = 0

print("Shakespeare infinito iniciado! Ctrl+C para parar.")

try:
    with open("shakespeare_infinito.txt", "a", encoding="utf-8") as f:
        while True:
            with torch.no_grad():
                generated = model.generate(
                    context,
                    max_new_tokens=1000
                )

            # Pega apenas os caracteres recém-gerados
            new_tokens = generated[0, context.shape[1]:].tolist()
            new_text = decode(new_tokens)

            f.write(new_text)
            f.flush()

            total_chars += len(new_text)
            print(f"Gerados {total_chars:,} caracteres")

            # Preserva o contexto recente
            context = generated[:, -block_size:]

except KeyboardInterrupt:
    print(f"\nGeração encerrada. Total: {total_chars:,} caracteres.")
