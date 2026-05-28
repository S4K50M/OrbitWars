from kaggle_environments import make

env = make("orbit_wars", configuration={"seed": 42}, debug=True)
env.run(["./agents/a-nearest-sniper.py", "random"])

# View result
final = env.steps[-1]
for i, s in enumerate(final):
    print(f"Player {i}: reward={s.reward}, status={s.status}")

# Render in a notebook
# Render to an HTML string
html_output = env.render(mode="html", width=800, height=600)

# Save it to a file
with open("replay.html", "w") as f:
    f.write(html_output)
    
print("Replay saved to replay.html")

