import math

# Represent a hole in the board (position), board or colored home
class BoardPosition:
    def __init__(self, q, r, spacing, postype='board'):
        # Axial coordinates (q, r) for hex grid; s = -q - r 
        self.q = q 
        self.r = r 

        # Convert to pixel coordinates for Tk display (pointy-top hexes)
        self.x = spacing * (math.sqrt(3) * q + math.sqrt(3)/2 * r)
        self.y = spacing * (3/2 * r)

        # Type of hole:'board' for regular holes, or color name for home triangles
        self.postype = postype  # default = 'board'
        self.occupied = False # Occupation by pin


class HexBoard:
    def __init__(self, R=4, hole_radius=18, spacing=34):
        self.R = R # Radius of the hexagon
        self.hole_radius = hole_radius # Circle radius for holes (for Tk display)
        self.spacing = spacing # Distance between centers of adjacent holes (for Tk display)

        # Mapping opposite home triangles (start with target home)
        self.colour_opposites ={'red':'blue', 'lawn green':'gray0', 'blue':'red', 'yellow':'purple', 'purple':'yellow', 'gray0':'lawn green'}

        # Axial coordinates (q, r), with s = -q - r
        self.cells = []                  # list of (q, r)
        self.hole_index_of = {}          # map (q, r) -> index
        self.cartesian = []              # list of (x, y) pixel coords (for Tk)
        self._rows = []                  # rows grouped by r for ASCII

        self._generate_hexagon()
        self._project_to_pixels()
        self._build_rows_for_ascii()

    def _generate_hexagon(self):
        # Generate a regular hexagon of radius R in axial coordinates.
        R = self.R
        cells = []

        # Generate axial coordinates for a hexagon of radius R
        for q in range(-R, R + 1):
            for r in range(-R, R + 1):
                s = -q - r # Calculate s coordinate for hex grid
                if max(abs(q), abs(r), abs(s)) <= R:
                    # Only include cells that are within the hexagon radius
                    newcell = BoardPosition(q, r, self.spacing)
                    cells.append(newcell)
                    #cells.append((q, r, 'p'))

        # Add colored home triangles (10 holes each) at the corners of the hexagon
        base_blue =[(1,-5),(2,-5),(3,-5),(4,-5), (2,-6),(3,-6),(4,-6), (3,-7),(4,-7), (4,-8)]
        base_red =[(-1,5),(-2,5),(-3,5), (-4,5),(-2,6),(-3,6),(-4,6),(-3,7),(-4,7),(-4,8)]
        base_yellow =[(-1,-4),(-2,-3),(-3,-2),(-4,-1), (-2,-4),(-3,-3),(-4,-2), (-3,-4), (-4,-3), (-4,-4) ]
        base_green =[(5,-4),(5,-3),(5,-2),(5,-1), (6,-4),(6,-3),(6,-2), (7,-4),(7,-3), (8,-4)]
        base_purple =[(1,4),(2,3),(3,2),(4,1), (2,4),(3,3),(4,2), (3,4),(4,3), (4,4)]
        base_gray0 =[(-5,1),(-5,2),(-5,3),(-5,4), (-6,2),(-6,3),(-6,4), (-7,3),(-7,4), (-8,4)]

        # Create BoardPosition objects for each cell, with postype indicating color of home or 'board' for regular holes
        for (q,r) in base_blue:
            newcell = BoardPosition(q, r, self.spacing, postype='blue')
            cells.append(newcell)
        for (q,r) in base_red:
            newcell = BoardPosition(q, r, self.spacing, postype='red')
            cells.append(newcell)
        for (q,r) in base_yellow:
            newcell = BoardPosition(q, r, self.spacing, postype='yellow')
            cells.append(newcell)
        for (q,r) in base_green:
            newcell = BoardPosition(q, r, self.spacing, postype='lawn green')
            cells.append(newcell)
        for (q,r) in base_purple:
            newcell = BoardPosition(q, r, self.spacing, postype='purple')
            cells.append(newcell)
        for (q,r) in base_gray0:
            newcell = BoardPosition(q, r, self.spacing, postype='gray0')
            cells.append(newcell)

        # Sort cells by r, then q for consistent ordering; build hole_index_of mapping
        cells.sort(key=lambda t: (t.r, t.q))
        self.cells = cells
        self.hole_index_of = {(ax.q,ax.r): i for i, ax in enumerate(cells)}
        #print('index',self.hole_index_of)

    # Convert axial coordinates to pixel coordinates for Tk display
    def _project_to_pixels(self):
        cart = []

        for t in self.cells:
            x = t.x
            y = t.y
            cart.append((x, y))
            # print(f"Cell (q={t.q}, r={t.r}) -> (x={x}, y={y}), {self.spacing}* {t.q} + {t.r}, {t.postype}")
        
        self.cartesian = cart

    # Group cells by r row for ASCII rendering; sort rows by r, and within each row by q
    def _build_rows_for_ascii(self):
        rows = {}

        for t in self.cells:
            rows.setdefault(t.r, []).append((t.q, t.r, t.postype))
        
        ordered = []
        for rr in sorted(rows.keys()):
            ordered.append(sorted(rows[rr], key=lambda x: x[0]))
        
        self._rows = ordered

    def print_ascii(self, pins=None, empty='·'):
        # Print the board as ASCII.
        pin_map = {}

        if pins:
            for p in pins:
                q= self.cells[p.axialindex].q
                r= self.cells[p.axialindex].r
                #q, r = self.cells[p.index]
                pin_map[(q, r)] = (p.color[:1].upper() if p.color else 'X')

        max_width = max(len(row) for row in self._rows)  # for indentation
        
        for row in self._rows:
            pad = " " * (max_width - len(row))  # left indentation to form a hex outline
            parts = []
            for (q, r, t) in row:
                parts.append(pin_map.get((q, r), empty if t == 'board' else t[:1].lower()))
            print(pad + " ".join(parts))

    # Board lookup helpers
    def axial_index(self, q, r):
        return self.hole_index_of[(q, r)]

    def axial_of_index(self, idx):
        return self.cells[idx]

    def axial_of_colour(self, colour):
        l = [(cell.q, cell.r) for cell in self.cells if cell.postype == colour]
        return [self.hole_index_of[(q,r)] for (q,r) in l]






